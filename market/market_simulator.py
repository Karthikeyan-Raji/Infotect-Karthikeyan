# ==============================================================================
# ChronosMatch: Zero-Copy High-Frequency Trading Engine
# Module: market/market_simulator.py
# Role: Person 2 (Ingestion, UI & Auditing Architect)
# Description: High-throughput asyncio order generator streaming 100,000 ticks/sec
#              into the POSIX shared memory ring buffer via struct.pack.
#              Simulates Geometric Brownian Motion, Poisson bursts, and Institutional Whale orders.
# ==============================================================================

import asyncio
import time
import random
import math
import struct
import os
import sys
from typing import Optional

# Ensure scratch/chronosmatch is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine import SPSCBufferWriter, TICK_STRUCT_FORMAT, TICK_STRUCT_SIZE

class MarketSimulator:
    """
    High-Frequency Order Flow Ingestion Generator.
    Pushes high-density structured binary ticks directly into shared memory.
    """
    def __init__(
        self,
        shm_path: str,
        target_rate: int = 100_000,
        initial_price: float = 150.00,
        volatility: float = 0.0002,
        whale_threshold: int = 5_000
    ):
        self.shm_path = shm_path
        self.target_rate = target_rate
        self.mid_price = initial_price
        self.volatility = volatility
        self.whale_threshold = whale_threshold

        self.writer = SPSCBufferWriter(shm_path, capacity=1048576)
        self.writer.init_and_map()

        self.current_order_id = 1
        self.is_running = False
        self.total_generated = 0
        self.whale_orders_sent = 0

    def generate_tick(self) -> tuple:
        """
        Generates a synthetic HFT tick adhering to microstructural realities:
        - Price drift via random walk
        - Bid/Ask limit spread tight to mid
        - Market order sweeps
        - Institutional Whale order spikes
        """
        self.current_order_id += 1
        order_id = self.current_order_id
        timestamp_ns = time.perf_counter_ns()

        # Asset price random walk
        drift = self.volatility * (random.random() - 0.5) * self.mid_price
        self.mid_price = max(1.0, round(self.mid_price + drift, 2))

        # Side: 50% Buy, 50% Sell
        side = 'B' if random.random() < 0.50 else 'S'

        # Probability distribution: 80% Limit, 17% Market, 3% Whale Block
        rand_role = random.random()
        if rand_role < 0.80:
            order_type = 'L'
            # Spread offset: +/- $0.01 to $0.05
            offset = round(random.uniform(0.01, 0.05), 2)
            price = round(self.mid_price - offset if side == 'B' else self.mid_price + offset, 2)
            qty = random.randint(10, 500)
        elif rand_role < 0.97:
            order_type = 'M'
            # Market orders cross spread aggressively
            price = round(self.mid_price + 0.10 if side == 'B' else self.mid_price - 0.10, 2)
            qty = random.randint(50, 800)
        else:
            # Whale Order Alert candidate
            order_type = 'L' if random.random() < 0.6 else 'M'
            price = round(self.mid_price, 2)
            qty = random.randint(self.whale_threshold, self.whale_threshold * 4)
            self.whale_orders_sent += 1

        return (order_id, timestamp_ns, price, qty, side, order_type)

    async def stream_orders(self, total_orders: int = 500_000, batch_size: int = 1_000):
        """
        Async streaming loop pushing orders in calibrated micro-batches
        to hit 100,000 orders/sec while yielding cooperatively to asyncio.
        """
        self.is_running = True
        self.total_generated = 0
        start_time = time.perf_counter()

        # Batch delay calculation to achieve target_rate
        # e.g., 1,000 orders per batch at 100k/s = 100 batches/sec -> 10ms per batch
        target_batch_dt = batch_size / self.target_rate

        while self.is_running and self.total_generated < total_orders:
            t0 = time.perf_counter()
            remaining = total_orders - self.total_generated
            current_batch_size = min(batch_size, remaining)

            # Rapid zero-copy batch push
            for _ in range(current_batch_size):
                order_id, ts, price, qty, side, o_type = self.generate_tick()
                success = self.writer.write_order(order_id, ts, price, qty, side, o_type)
                if not success:
                    # Ring buffer backpressure; yield briefly
                    await asyncio.sleep(0.0001)
                    self.writer.write_order(order_id, ts, price, qty, side, o_type)
                self.total_generated += 1

            elapsed = time.perf_counter() - t0
            sleep_needed = target_batch_dt - elapsed
            if sleep_needed > 0.0005:
                await asyncio.sleep(sleep_needed)
            else:
                # Yield control cooperatively to event loop
                await asyncio.sleep(0)

        total_elapsed = time.perf_counter() - start_time
        rate = self.total_generated / total_elapsed if total_elapsed > 0 else 0
        self.is_running = False
        return {
            "total_orders": self.total_generated,
            "whale_orders": self.whale_orders_sent,
            "elapsed_seconds": total_elapsed,
            "effective_rate_ops": rate
        }

    def close(self):
        self.is_running = False
        self.writer.close()


async def main():
    shm_file = "/dev/shm/chronos_orders.dat" if os.path.exists("/dev/shm") else os.path.abspath("chronos_orders.dat")
    print(f"[*] Starting Market Simulator on {shm_file}")
    sim = MarketSimulator(shm_file, target_rate=100_000)
    print("[*] Generating 200,000 orders at 100,000 ticks/sec...")
    res = await sim.stream_orders(total_orders=200_000)
    print(f"[+] Complete: {res['total_orders']:,} orders in {res['elapsed_seconds']:.2f}s ({res['effective_rate_ops']:,.0f} ticks/s)")
    sim.close()

if __name__ == "__main__":
    asyncio.run(main())
