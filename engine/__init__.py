# ==============================================================================
# ChronosMatch: Zero-Copy High-Frequency Trading Engine
# Module: engine/__init__.py
# Role:Person 1 (Low-Latency Core & Memory Architect)
# Description: Engine module loader with seamless Cython compiled speedup and
#              pure-Python/ctypes fallback for zero-dependency execution.
# ==============================================================================

import struct
import time
import os
import mmap
from collections import deque

__all__ = ["SPSCBufferReader", "SPSCBufferWriter", "MatchingEngineCore", "TICK_STRUCT_FORMAT", "TICK_STRUCT_SIZE"]

# Binary Contract: 32 bytes packed '<QQdIcc2x'
TICK_STRUCT_FORMAT = "<QQdIcc2x"
TICK_STRUCT_SIZE = struct.calcsize(TICK_STRUCT_FORMAT)
assert TICK_STRUCT_SIZE == 32, f"OrderTick struct must be 32 bytes, got {TICK_STRUCT_SIZE}"

# Header layout: head (8B), pad1 (56B), tail (8B), pad2 (56B), capacity (8B), mask (8B), pad3 (48B) = 192 bytes
HEADER_FORMAT = "<Q56sQ56sQQ48s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_SIZE == 192, f"RingBufferHeader struct must be 192 bytes, got {HEADER_SIZE}"

# Attempt to load compiled Cython modules first
try:
    from engine.ipc_ring_buffer import SPSCBufferReader, SPSCBufferWriter
    from engine.matching_engine import MatchingEngineCore
    CYTHON_ACCELERATED = True
except ImportError:
    CYTHON_ACCELERATED = False

if not CYTHON_ACCELERATED:
    # --------------------------------------------------------------------------
    # Pure-Python / Ctypes Compatible Fallback Implementation
    # Exact same 32-byte binary protocol, zero-copy layout, and nogil-style LOB
    # --------------------------------------------------------------------------
    class SPSCBufferWriter:
        def __init__(self, file_path: str, capacity: int = 1048576):
            self.file_path = file_path
            self.capacity = capacity
            self.mask = capacity - 1
            self.header_size = HEADER_SIZE
            self.slot_size = TICK_STRUCT_SIZE
            self.total_size = self.header_size + (self.capacity * self.slot_size)
            self.mmap_obj = None

        def init_and_map(self):
            parent = os.path.dirname(self.file_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)

            with open(self.file_path, "wb") as f:
                f.seek(self.total_size - 1)
                f.write(b"\0")
                f.flush()

            f = open(self.file_path, "r+b")
            self.mmap_obj = mmap.mmap(f.fileno(), self.total_size, access=mmap.ACCESS_WRITE)
            f.close()

            # Initialize header: head=0, tail=0, capacity, mask
            header_bytes = struct.pack(
                HEADER_FORMAT,
                0, b"\0" * 56,
                0, b"\0" * 56,
                self.capacity, self.mask,
                b"\0" * 48
            )
            self.mmap_obj[:self.header_size] = header_bytes

        def write_order(self, order_id: int, timestamp_ns: int, price: float,
                        qty: int, side: str, order_type: str) -> bool:
            # Read head & tail
            head = struct.unpack_from("<Q", self.mmap_obj, 0)[0]
            tail = struct.unpack_from("<Q", self.mmap_obj, 64)[0]

            if (head - tail) >= self.capacity:
                return False  # Buffer full

            slot_idx = head & self.mask
            offset = self.header_size + (slot_idx * self.slot_size)

            struct.pack_into(
                TICK_STRUCT_FORMAT,
                self.mmap_obj,
                offset,
                order_id,
                timestamp_ns,
                float(price),
                int(qty),
                side.encode('ascii')[0:1],
                order_type.encode('ascii')[0:1]
            )

            # Atomic increment write index
            struct.pack_into("<Q", self.mmap_obj, 0, head + 1)
            return True

        def close(self):
            if self.mmap_obj is not None:
                self.mmap_obj.close()
                self.mmap_obj = None

    class SPSCBufferReader:
        def __init__(self, file_path: str):
            self.file_path = file_path
            self.header_size = HEADER_SIZE
            self.slot_size = TICK_STRUCT_SIZE
            self.mmap_obj = None
            self.capacity = 0
            self.mask = 0

        def map_buffer(self):
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Shared memory file not found: {self.file_path}")

            f = open(self.file_path, "r+b")
            self.mmap_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
            f.close()

            # Read capacity and mask from header
            head, _, tail, _, cap, mask, _ = struct.unpack_from(HEADER_FORMAT, self.mmap_obj, 0)
            self.capacity = cap
            self.mask = mask

        def read_one(self):
            head = struct.unpack_from("<Q", self.mmap_obj, 0)[0]
            tail = struct.unpack_from("<Q", self.mmap_obj, 64)[0]

            if tail >= head:
                return None

            slot_idx = tail & self.mask
            offset = self.header_size + (slot_idx * self.slot_size)

            order_id, ts, price, qty, side_b, type_b = struct.unpack_from(
                TICK_STRUCT_FORMAT, self.mmap_obj, offset
            )

            # Advance tail
            struct.pack_into("<Q", self.mmap_obj, 64, tail + 1)

            return {
                "order_id": order_id,
                "timestamp_ns": ts,
                "price": round(price, 4),
                "qty": qty,
                "side": side_b.decode('ascii'),
                "order_type": type_b.decode('ascii'),
            }

        def close(self):
            if self.mmap_obj is not None:
                self.mmap_obj.close()
                self.mmap_obj = None

    class MatchingEngineCore:
        def __init__(self):
            self.bids = {}  # price -> deque of order dicts
            self.asks = {}  # price -> deque of order dicts
            self.sorted_bid_prices = []  # descending
            self.sorted_ask_prices = []  # ascending
            self.fills = []
            self.latencies = []
            self.total_orders = 0
            self.total_volume = 0
            self.match_seq = 0

        def process_tick(self, tick: dict):
            t_enter = time.perf_counter_ns()
            self.total_orders += 1

            side = tick["side"]
            order_type = tick["order_type"]
            price = tick["price"]
            qty = tick["qty"]
            order_id = tick["order_id"]
            ts = tick["timestamp_ns"]

            remaining_qty = qty

            if order_type == 'C':
                return

            if side == 'B':
                while remaining_qty > 0 and self.sorted_ask_prices:
                    best_ask = self.sorted_ask_prices[0]
                    if order_type != 'M' and price < best_ask:
                        break

                    ask_queue = self.asks[best_ask]
                    while remaining_qty > 0 and ask_queue:
                        book_order = ask_queue[0]
                        fill_qty = min(remaining_qty, book_order["qty"])
                        remaining_qty -= fill_qty
                        book_order["qty"] -= fill_qty
                        self.total_volume += fill_qty

                        lat = time.perf_counter_ns() - ts
                        self.fills.append((
                            self.match_seq,
                            order_id,
                            book_order["order_id"],
                            best_ask,
                            fill_qty,
                            lat
                        ))
                        self.match_seq += 1

                        if book_order["qty"] == 0:
                            ask_queue.popleft()

                    if not ask_queue:
                        del self.asks[best_ask]
                        self.sorted_ask_prices.pop(0)

                if order_type == 'L' and remaining_qty > 0:
                    if price not in self.bids:
                        self.bids[price] = deque()
                        self.sorted_bid_prices.append(price)
                        self.sorted_bid_prices.sort(reverse=True)
                    self.bids[price].append({
                        "order_id": order_id,
                        "price": price,
                        "qty": remaining_qty,
                        "timestamp_ns": ts
                    })

            else:  # Sell
                while remaining_qty > 0 and self.sorted_bid_prices:
                    best_bid = self.sorted_bid_prices[0]
                    if order_type != 'M' and price > best_bid:
                        break

                    bid_queue = self.bids[best_bid]
                    while remaining_qty > 0 and bid_queue:
                        book_order = bid_queue[0]
                        fill_qty = min(remaining_qty, book_order["qty"])
                        remaining_qty -= fill_qty
                        book_order["qty"] -= fill_qty
                        self.total_volume += fill_qty

                        lat = time.perf_counter_ns() - ts
                        self.fills.append((
                            self.match_seq,
                            book_order["order_id"],
                            order_id,
                            best_bid,
                            fill_qty,
                            lat
                        ))
                        self.match_seq += 1

                        if book_order["qty"] == 0:
                            bid_queue.popleft()

                    if not bid_queue:
                        del self.bids[best_bid]
                        self.sorted_bid_prices.pop(0)

                if order_type == 'L' and remaining_qty > 0:
                    if price not in self.asks:
                        self.asks[price] = deque()
                        self.sorted_ask_prices.append(price)
                        self.sorted_ask_prices.sort()
                    self.asks[price].append({
                        "order_id": order_id,
                        "price": price,
                        "qty": remaining_qty,
                        "timestamp_ns": ts
                    })

            lat_total = time.perf_counter_ns() - t_enter
            self.latencies.append(lat_total)

        def run_consumer_loop(self, reader, max_orders=0, stop_on_empty=False):
            processed = 0
            while True:
                tick = reader.read_one()
                if tick is not None:
                    self.process_tick(tick)
                    processed += 1
                    if max_orders > 0 and processed >= max_orders:
                        break
                else:
                    if stop_on_empty and processed > 0:
                        break
                    time.sleep(0.0001)
            return processed

        def get_market_state(self):
            best_bid = self.sorted_bid_prices[0] if self.sorted_bid_prices else 0.0
            bid_vol = sum(o["qty"] for o in self.bids[best_bid]) if best_bid else 0
            best_ask = self.sorted_ask_prices[0] if self.sorted_ask_prices else 0.0
            ask_vol = sum(o["qty"] for o in self.asks[best_ask]) if best_ask else 0
            spread = (best_ask - best_bid) if (best_bid and best_ask) else 0.0

            bid_depth = []
            for p in self.sorted_bid_prices[:5]:
                bid_depth.append((p, sum(o["qty"] for o in self.bids[p])))

            ask_depth = []
            for p in self.sorted_ask_prices[:5]:
                ask_depth.append((p, sum(o["qty"] for o in self.asks[p])))

            return {
                "best_bid": best_bid,
                "bid_vol": bid_vol,
                "best_ask": best_ask,
                "ask_vol": ask_vol,
                "spread": spread,
                "bids": bid_depth,
                "asks": ask_depth,
                "total_orders": self.total_orders,
                "total_volume": self.total_volume,
                "total_fills": len(self.fills)
            }

        def drain_fills(self, start_idx=0):
            return self.fills[start_idx:]

        def get_latency_stats(self):
            if not self.latencies:
                return {"count": 0, "p50_us": 0.0, "p90_us": 0.0, "p99_us": 0.0, "p99_9_us": 0.0, "max_us": 0.0, "mean_us": 0.0}

            us_list = [ns / 1000.0 for ns in self.latencies]
            us_list.sort()
            n = len(us_list)
            return {
                "count": n,
                "p50_us": us_list[int(n * 0.50)],
                "p90_us": us_list[int(n * 0.90)],
                "p99_us": us_list[int(n * 0.99)],
                "p99_9_us": us_list[int(n * 0.999)],
                "max_us": us_list[-1],
                "mean_us": sum(us_list) / n
            }
