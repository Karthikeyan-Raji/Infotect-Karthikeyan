# ==============================================================================
# ChronosMatch: Zero-Copy High-Frequency Trading Engine
# Module: storage/ledger_writer.py
# Role: Person 2 (Ingestion, UI & Auditing Architect)
# Description: Asynchronous trade ledger writer draining execution fills into
#              an optimized SQLite database (trades.db) via WAL mode and batched commits.
# ==============================================================================

import asyncio
import sqlite3
import time
import os
from typing import List, Tuple

class AsyncLedgerWriter:
    """
    High-throughput persistence worker for audit logging and trade settlement.
    Uses SQLite in Write-Ahead Logging (WAL) mode with batch transactions
    to avoid blocking low-latency execution paths.
    """
    def __init__(self, db_path: str = "trades.db", batch_size: int = 1_000):
        self.db_path = db_path
        self.batch_size = batch_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=100_000)
        self.is_running = False
        self.total_persisted = 0
        self.conn = None

        self._init_database()

    def _init_database(self):
        """Initializes SQLite schema and performance PRAGMAs."""
        parent = os.path.dirname(self.db_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = self.conn.cursor()

        # Low-latency SQLite optimization pragmas
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA cache_size = -64000;")  # 64MB memory cache
        cur.execute("PRAGMA temp_store = MEMORY;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                match_id INTEGER PRIMARY KEY,
                buy_order_id INTEGER NOT NULL,
                sell_order_id INTEGER NOT NULL,
                fill_price REAL NOT NULL,
                fill_qty INTEGER NOT NULL,
                latency_ns INTEGER NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_buy_order ON trades(buy_order_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sell_order ON trades(sell_order_id);")
        self.conn.commit()

    async def enqueue_fills(self, fills: List[Tuple]):
        """Non-blocking ingestion of matched fill tuples from execution bus."""
        for f in fills:
            try:
                self.queue.put_nowait(f)
            except asyncio.QueueFull:
                await self.queue.put(f)

    async def start_worker(self):
        """Asynchronous batch drainer loop."""
        self.is_running = True
        batch = []
        cur = self.conn.cursor()

        while self.is_running:
            try:
                # Wait for items or flush on timeout
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=0.05)
                    batch.append(item)
                    self.queue.task_done()
                except asyncio.TimeoutError:
                    pass

                # Drain all available items up to batch_size
                while not self.queue.empty() and len(batch) < self.batch_size:
                    batch.append(self.queue.get_nowait())
                    self.queue.task_done()

                # Flush batch to disk inside single transaction
                if batch:
                    cur.execute("BEGIN TRANSACTION;")
                    cur.executemany("""
                        INSERT OR IGNORE INTO trades (
                            match_id, buy_order_id, sell_order_id, fill_price, fill_qty, latency_ns
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, batch)
                    self.conn.commit()
                    self.total_persisted += len(batch)
                    batch.clear()

            except Exception as e:
                # In production HFT, log error to secondary monitoring ring
                await asyncio.sleep(0.01)

    async def flush_and_close(self):
        """Flushes residual queue items and closes database connection."""
        self.is_running = False
        cur = self.conn.cursor()
        batch = []
        while not self.queue.empty():
            batch.append(self.queue.get_nowait())
            self.queue.task_done()

        if batch:
            cur.execute("BEGIN TRANSACTION;")
            cur.executemany("""
                INSERT OR IGNORE INTO trades (
                    match_id, buy_order_id, sell_order_id, fill_price, fill_qty, latency_ns
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, batch)
            self.conn.commit()
            self.total_persisted += len(batch)

        if self.conn:
            self.conn.close()
            self.conn = None
