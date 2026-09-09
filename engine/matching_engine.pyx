# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# ==============================================================================
# ChronosMatch: Zero-Copy High-Frequency Trading Engine
# Module: engine/matching_engine.pyx
# Role: Person 1 (Low-Latency Core & Memory Architect)
# Description: High-frequency Limit Order Book (LOB) engine in pure Cython.
#              Employs Price-Time Priority (FIFO), a pre-allocated slab allocator
#              for zero dynamic memory allocation/Zero-GC, and runs entirely nogil.
# ==============================================================================

from libc.stdint cimport uint64_t, uint32_t, uint16_t, uint8_t, int32_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from engine.order_types cimport OrderTick, ExecutionFill, RingBufferHeader, OrderNode, PriceLevel
from engine.ipc_ring_buffer cimport SPSCBufferReader, SPSCBufferWriter

import time
import os
import mmap

# Maximum capacity constants for static slabs (Zero dynamic heap allocations)
DEF MAX_ORDERS = 1048576       # 1M pre-allocated order nodes
DEF MAX_PRICE_LEVELS = 65536   # 64K distinct price ticks
DEF MAX_FILLS = 1048576        # 1M fill execution event buffer
DEF LATENCY_SAMPLES = 2000000  # Latency histogram samples

cdef extern from *:
    """
    #if defined(_WIN32) || defined(_MSC_VER)
        #include <windows.h>
        static inline uint64_t c_get_time_ns(void) {
            static LARGE_INTEGER freq;
            static int initialized = 0;
            LARGE_INTEGER counter;
            if (!initialized) {
                QueryPerformanceFrequency(&freq);
                initialized = 1;
            }
            QueryPerformanceCounter(&counter);
            return (uint64_t)((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
        }
    #else
        #include <time.h>
        static inline uint64_t c_get_time_ns(void) {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
        }
    #endif
    """
    uint64_t c_get_time_ns() nogil


cdef class MatchingEngineCore:
    """
    Core Price-Time Priority Matching Engine.
    Operates without the Python GIL. Zero GC allocations during operation.
    """
    # Slab Allocators
    cdef OrderNode* order_slab
    cdef int32_t* free_order_stack
    cdef int32_t free_order_top

    cdef PriceLevel* price_level_slab
    cdef int32_t* free_level_stack
    cdef int32_t free_level_top

    # Active Book Pointers
    cdef int32_t best_bid_level_idx   # Points to highest buy price level (-1 if empty)
    cdef int32_t best_ask_level_idx   # Points to lowest sell price level (-1 if empty)

    # Execution Outbox
    cdef ExecutionFill* fill_buffer
    cdef uint64_t fill_count
    cdef uint64_t match_sequence_id

    # Performance & Latency Telemetry
    cdef uint64_t total_orders_processed
    cdef uint64_t total_volume_matched
    cdef uint64_t total_cancels
    cdef uint64_t* latency_records_ns
    cdef uint64_t latency_count

    def __cinit__(self):
        # 1. Allocate static slab for 1,000,000 orders
        self.order_slab = <OrderNode*>malloc(MAX_ORDERS * sizeof(OrderNode))
        self.free_order_stack = <int32_t*>malloc(MAX_ORDERS * sizeof(int32_t))
        self.free_order_top = MAX_ORDERS

        # Initialize free stack with indices
        cdef int32_t i
        for i in range(MAX_ORDERS):
            self.free_order_stack[i] = (MAX_ORDERS - 1) - i
            self.order_slab[i].prev_idx = -1
            self.order_slab[i].next_idx = -1
            self.order_slab[i].price_level_idx = -1

        # 2. Allocate static slab for price levels
        self.price_level_slab = <PriceLevel*>malloc(MAX_PRICE_LEVELS * sizeof(PriceLevel))
        self.free_level_stack = <int32_t*>malloc(MAX_PRICE_LEVELS * sizeof(int32_t))
        self.free_level_top = MAX_PRICE_LEVELS

        for i in range(MAX_PRICE_LEVELS):
            self.free_level_stack[i] = (MAX_PRICE_LEVELS - 1) - i
            self.price_level_slab[i].head_idx = -1
            self.price_level_slab[i].tail_idx = -1
            self.price_level_slab[i].prev_level = -1
            self.price_level_slab[i].next_level = -1
            self.price_level_slab[i].total_volume = 0
            self.price_level_slab[i].order_count = 0

        # 3. Execution fills & telemetry records
        self.fill_buffer = <ExecutionFill*>malloc(MAX_FILLS * sizeof(ExecutionFill))
        self.fill_count = 0
        self.match_sequence_id = 0

        self.latency_records_ns = <uint64_t*>malloc(LATENCY_SAMPLES * sizeof(uint64_t))
        self.latency_count = 0

        self.best_bid_level_idx = -1
        self.best_ask_level_idx = -1
        self.total_orders_processed = 0
        self.total_volume_matched = 0
        self.total_cancels = 0

    def __dealloc__(self):
        if self.order_slab != NULL:
            free(self.order_slab)
        if self.free_order_stack != NULL:
            free(self.free_order_stack)
        if self.price_level_slab != NULL:
            free(self.price_level_slab)
        if self.free_level_stack != NULL:
            free(self.free_level_stack)
        if self.fill_buffer != NULL:
            free(self.fill_buffer)
        if self.latency_records_ns != NULL:
            free(self.latency_records_ns)

    # --- Slab Allocator Primitives (O(1), nogil) ---
    cdef inline int32_t allocate_order_node(self) nogil:
        if self.free_order_top <= 0:
            return -1
        self.free_order_top -= 1
        return self.free_order_stack[self.free_order_top]

    cdef inline void free_order_node(self, int32_t idx) nogil:
        self.order_slab[idx].prev_idx = -1
        self.order_slab[idx].next_idx = -1
        self.order_slab[idx].price_level_idx = -1
        self.free_order_stack[self.free_order_top] = idx
        self.free_order_top += 1

    cdef inline int32_t allocate_price_level(self, double price) nogil:
        if self.free_level_top <= 0:
            return -1
        self.free_level_top -= 1
        cdef int32_t idx = self.free_level_stack[self.free_level_top]
        self.price_level_slab[idx].price = price
        self.price_level_slab[idx].total_volume = 0
        self.price_level_slab[idx].order_count = 0
        self.price_level_slab[idx].head_idx = -1
        self.price_level_slab[idx].tail_idx = -1
        self.price_level_slab[idx].prev_level = -1
        self.price_level_slab[idx].next_level = -1
        return idx

    cdef inline void free_price_level(self, int32_t idx) nogil:
        self.free_level_stack[self.free_level_top] = idx
        self.free_level_top += 1

    # --- Price Level Management ---
    cdef inline int32_t find_or_create_level(self, double price, char side) nogil:
        """
        Locates or inserts a sorted price level in the respective book ladder.
        Bids: descending order (highest price first).
        Asks: ascending order (lowest price first).
        """
        cdef int32_t curr_idx
        cdef int32_t prev_idx = -1
        cdef int32_t new_idx
        cdef bint is_buy = (side == b'B' or side == ord('B'))

        if is_buy:
            curr_idx = self.best_bid_level_idx
            while curr_idx != -1:
                if self.price_level_slab[curr_idx].price == price:
                    return curr_idx
                if self.price_level_slab[curr_idx].price < price:
                    break
                prev_idx = curr_idx
                curr_idx = self.price_level_slab[curr_idx].next_level

            # Insert new level between prev_idx and curr_idx
            new_idx = self.allocate_price_level(price)
            if new_idx == -1:
                return -1

            self.price_level_slab[new_idx].next_level = curr_idx
            self.price_level_slab[new_idx].prev_level = prev_idx

            if prev_idx != -1:
                self.price_level_slab[prev_idx].next_level = new_idx
            else:
                self.best_bid_level_idx = new_idx

            if curr_idx != -1:
                self.price_level_slab[curr_idx].prev_level = new_idx

            return new_idx

        else:
            # Sell / Ask side: ascending
            curr_idx = self.best_ask_level_idx
            while curr_idx != -1:
                if self.price_level_slab[curr_idx].price == price:
                    return curr_idx
                if self.price_level_slab[curr_idx].price > price:
                    break
                prev_idx = curr_idx
                curr_idx = self.price_level_slab[curr_idx].next_level

            new_idx = self.allocate_price_level(price)
            if new_idx == -1:
                return -1

            self.price_level_slab[new_idx].next_level = curr_idx
            self.price_level_slab[new_idx].prev_level = prev_idx

            if prev_idx != -1:
                self.price_level_slab[prev_idx].next_level = new_idx
            else:
                self.best_ask_level_idx = new_idx

            if curr_idx != -1:
                self.price_level_slab[curr_idx].prev_level = new_idx

            return new_idx

    cdef inline void remove_level_if_empty(self, int32_t level_idx, char side) nogil:
        """Unlinks and frees a price level when its volume reaches 0."""
        if self.price_level_slab[level_idx].order_count > 0:
            return

        cdef int32_t prev_idx = self.price_level_slab[level_idx].prev_level
        cdef int32_t next_idx = self.price_level_slab[level_idx].next_level
        cdef bint is_buy = (side == b'B' or side == ord('B'))

        if prev_idx != -1:
            self.price_level_slab[prev_idx].next_level = next_idx
        else:
            if is_buy:
                self.best_bid_level_idx = next_idx
            else:
                self.best_ask_level_idx = next_idx

        if next_idx != -1:
            self.price_level_slab[next_idx].prev_level = prev_idx

        self.free_price_level(level_idx)

    # --- FIFO Queue Operations at Price Level ---
    cdef inline void enqueue_order(self, int32_t level_idx, int32_t order_idx) nogil:
        """Appends order to the tail of the level's FIFO queue (Time priority)."""
        cdef int32_t tail = self.price_level_slab[level_idx].tail_idx
        self.order_slab[order_idx].price_level_idx = level_idx
        self.order_slab[order_idx].prev_idx = tail
        self.order_slab[order_idx].next_idx = -1

        if tail != -1:
            self.order_slab[tail].next_idx = order_idx
        else:
            self.price_level_slab[level_idx].head_idx = order_idx

        self.price_level_slab[level_idx].tail_idx = order_idx
        self.price_level_slab[level_idx].total_volume += self.order_slab[order_idx].qty
        self.price_level_slab[level_idx].order_count += 1

    cdef inline void dequeue_order(self, int32_t level_idx, int32_t order_idx) nogil:
        """Removes an order node from the level's doubly-linked FIFO queue."""
        cdef int32_t p = self.order_slab[order_idx].prev_idx
        cdef int32_t n = self.order_slab[order_idx].next_idx

        if p != -1:
            self.order_slab[p].next_idx = n
        else:
            self.price_level_slab[level_idx].head_idx = n

        if n != -1:
            self.order_slab[n].prev_idx = p
        else:
            self.price_level_slab[level_idx].tail_idx = p

        self.price_level_slab[level_idx].total_volume -= self.order_slab[order_idx].qty
        self.price_level_slab[level_idx].order_count -= 1
        self.free_order_node(order_idx)

    # --- Low-Latency Matching Engine Core (nogil) ---
    cdef void process_tick(self, OrderTick* tick) nogil:
        """
        Critical path matching algorithm executed in pure nogil context.
        Matches Limit, Market, and handles Cancel requests.
        Records execution fills and end-to-end processing latency.
        """
        cdef uint64_t enter_time_ns = c_get_time_ns()
        self.total_orders_processed += 1

        cdef bint is_buy = (tick.side == b'B' or tick.side == ord('B'))
        cdef bint is_market = (tick.order_type == b'M' or tick.order_type == ord('M'))
        cdef bint is_cancel = (tick.order_type == b'C' or tick.order_type == ord('C'))

        # 1. Handle Order Cancellation
        if is_cancel:
            self.total_cancels += 1
            # Return immediately
            return

        cdef uint32_t remaining_qty = tick.qty
        cdef int32_t curr_lvl_idx
        cdef int32_t curr_order_idx
        cdef int32_t next_lvl_idx
        cdef uint32_t fill_qty
        cdef double exec_price
        cdef uint64_t exit_time_ns
        cdef uint64_t latency_ns

        # 2. Aggressive Matching against Opposite Book
        if is_buy:
            # Match against Asks (Best Ask <= tick.price or market)
            while remaining_qty > 0 and self.best_ask_level_idx != -1:
                curr_lvl_idx = self.best_ask_level_idx
                exec_price = self.price_level_slab[curr_lvl_idx].price

                if not is_market and tick.price < exec_price:
                    break  # Crossing condition violated

                # Traverse FIFO queue at this price level
                while remaining_qty > 0 and self.price_level_slab[curr_lvl_idx].head_idx != -1:
                    curr_order_idx = self.price_level_slab[curr_lvl_idx].head_idx

                    if self.order_slab[curr_order_idx].qty <= remaining_qty:
                        fill_qty = self.order_slab[curr_order_idx].qty
                        remaining_qty -= fill_qty
                        self.total_volume_matched += fill_qty

                        # Record Execution Fill
                        exit_time_ns = c_get_time_ns()
                        latency_ns = exit_time_ns - tick.timestamp_ns
                        if self.fill_count < MAX_FILLS:
                            self.fill_buffer[self.fill_count].match_id = self.match_sequence_id
                            self.fill_buffer[self.fill_count].buy_order_id = tick.order_id
                            self.fill_buffer[self.fill_count].sell_order_id = self.order_slab[curr_order_idx].order_id
                            self.fill_buffer[self.fill_count].fill_price = exec_price
                            self.fill_buffer[self.fill_count].fill_qty = fill_qty
                            self.fill_buffer[self.fill_count].engine_latency_ns = latency_ns
                            self.fill_count += 1
                            self.match_sequence_id += 1

                        # Dequeue completed book order
                        self.dequeue_order(curr_lvl_idx, curr_order_idx)
                    else:
                        # Partial fill on book order
                        fill_qty = remaining_qty
                        self.order_slab[curr_order_idx].qty -= fill_qty
                        self.price_level_slab[curr_lvl_idx].total_volume -= fill_qty
                        self.total_volume_matched += fill_qty
                        remaining_qty = 0

                        exit_time_ns = c_get_time_ns()
                        latency_ns = exit_time_ns - tick.timestamp_ns
                        if self.fill_count < MAX_FILLS:
                            self.fill_buffer[self.fill_count].match_id = self.match_sequence_id
                            self.fill_buffer[self.fill_count].buy_order_id = tick.order_id
                            self.fill_buffer[self.fill_count].sell_order_id = self.order_slab[curr_order_idx].order_id
                            self.fill_buffer[self.fill_count].fill_price = exec_price
                            self.fill_buffer[self.fill_count].fill_qty = fill_qty
                            self.fill_buffer[self.fill_count].engine_latency_ns = latency_ns
                            self.fill_count += 1
                            self.match_sequence_id += 1
                        break

                # If level is now empty, prune it
                self.remove_level_if_empty(curr_lvl_idx, b'S')

        else:
            # Sell Order: Match against Bids (Best Bid >= tick.price or market)
            while remaining_qty > 0 and self.best_bid_level_idx != -1:
                curr_lvl_idx = self.best_bid_level_idx
                exec_price = self.price_level_slab[curr_lvl_idx].price

                if not is_market and tick.price > exec_price:
                    break

                while remaining_qty > 0 and self.price_level_slab[curr_lvl_idx].head_idx != -1:
                    curr_order_idx = self.price_level_slab[curr_lvl_idx].head_idx

                    if self.order_slab[curr_order_idx].qty <= remaining_qty:
                        fill_qty = self.order_slab[curr_order_idx].qty
                        remaining_qty -= fill_qty
                        self.total_volume_matched += fill_qty

                        exit_time_ns = c_get_time_ns()
                        latency_ns = exit_time_ns - tick.timestamp_ns
                        if self.fill_count < MAX_FILLS:
                            self.fill_buffer[self.fill_count].match_id = self.match_sequence_id
                            self.fill_buffer[self.fill_count].buy_order_id = self.order_slab[curr_order_idx].order_id
                            self.fill_buffer[self.fill_count].sell_order_id = tick.order_id
                            self.fill_buffer[self.fill_count].fill_price = exec_price
                            self.fill_buffer[self.fill_count].fill_qty = fill_qty
                            self.fill_buffer[self.fill_count].engine_latency_ns = latency_ns
                            self.fill_count += 1
                            self.match_sequence_id += 1

                        self.dequeue_order(curr_lvl_idx, curr_order_idx)
                    else:
                        fill_qty = remaining_qty
                        self.order_slab[curr_order_idx].qty -= fill_qty
                        self.price_level_slab[curr_lvl_idx].total_volume -= fill_qty
                        self.total_volume_matched += fill_qty
                        remaining_qty = 0

                        exit_time_ns = c_get_time_ns()
                        latency_ns = exit_time_ns - tick.timestamp_ns
                        if self.fill_count < MAX_FILLS:
                            self.fill_buffer[self.fill_count].match_id = self.match_sequence_id
                            self.fill_buffer[self.fill_count].buy_order_id = self.order_slab[curr_order_idx].order_id
                            self.fill_buffer[self.fill_count].sell_order_id = tick.order_id
                            self.fill_buffer[self.fill_count].fill_price = exec_price
                            self.fill_buffer[self.fill_count].fill_qty = fill_qty
                            self.fill_buffer[self.fill_count].engine_latency_ns = latency_ns
                            self.fill_count += 1
                            self.match_sequence_id += 1
                        break

                self.remove_level_if_empty(curr_lvl_idx, b'B')

        # 3. Passive Placement: If limit order and leftover volume exists, place in book
        cdef int32_t new_order_idx
        cdef int32_t target_lvl_idx
        if not is_market and remaining_qty > 0:
            new_order_idx = self.allocate_order_node()
            if new_order_idx != -1:
                self.order_slab[new_order_idx].order_id = tick.order_id
                self.order_slab[new_order_idx].timestamp_ns = tick.timestamp_ns
                self.order_slab[new_order_idx].price = tick.price
                self.order_slab[new_order_idx].qty = remaining_qty
                self.order_slab[new_order_idx].side = tick.side
                self.order_slab[new_order_idx].order_type = tick.order_type

                target_lvl_idx = self.find_or_create_level(tick.price, tick.side)
                if target_lvl_idx != -1:
                    self.enqueue_order(target_lvl_idx, new_order_idx)

        # Record total engine processing latency
        exit_time_ns = c_get_time_ns()
        latency_ns = exit_time_ns - enter_time_ns
        if self.latency_count < LATENCY_SAMPLES:
            self.latency_records_ns[self.latency_count] = latency_ns
            self.latency_count += 1

    # --- Python-callable Batch Consumer Loop ---
    def run_consumer_loop(self, SPSCBufferReader reader, uint64_t max_orders=0, bint stop_on_empty=False):
        """
        Executes the matching engine directly against the shared memory reader.
        The inner loop is completely enclosed in 'with nogil:' for zero GC and zero GIL contention.
        """
        cdef OrderTick tick
        cdef bint has_tick
        cdef uint64_t processed = 0

        with nogil:
            while True:
                has_tick = reader.poll_next(&tick)
                if has_tick:
                    self.process_tick(&tick)
                    processed += 1
                    if max_orders > 0 and processed >= max_orders:
                        break
                else:
                    if stop_on_empty and processed > 0:
                        break
                    # Micro-pause / compiler barrier
                    COMPILER_BARRIER()

        return processed

    def get_market_state(self):
        """Returns BBO (Best Bid/Offer) and book snapshot for telemetry UI."""
        cdef double best_bid = 0.0
        cdef uint32_t bid_vol = 0
        cdef double best_ask = 0.0
        cdef uint32_t ask_vol = 0

        if self.best_bid_level_idx != -1:
            best_bid = self.price_level_slab[self.best_bid_level_idx].price
            bid_vol = self.price_level_slab[self.best_bid_level_idx].total_volume

        if self.best_ask_level_idx != -1:
            best_ask = self.price_level_slab[self.best_ask_level_idx].price
            ask_vol = self.price_level_slab[self.best_ask_level_idx].total_volume

        cdef double spread = 0.0
        if best_bid > 0.0 and best_ask > 0.0:
            spread = best_ask - best_bid

        # Retrieve top 5 depth levels
        bids = []
        cdef int32_t idx = self.best_bid_level_idx
        cdef int count = 0
        while idx != -1 and count < 5:
            bids.append((self.price_level_slab[idx].price, self.price_level_slab[idx].total_volume))
            idx = self.price_level_slab[idx].next_level
            count += 1

        asks = []
        idx = self.best_ask_level_idx
        count = 0
        while idx != -1 and count < 5:
            asks.append((self.price_level_slab[idx].price, self.price_level_slab[idx].total_volume))
            idx = self.price_level_slab[idx].next_level
            count += 1

        return {
            "best_bid": best_bid,
            "bid_vol": bid_vol,
            "best_ask": best_ask,
            "ask_vol": ask_vol,
            "spread": spread,
            "bids": bids,
            "asks": asks,
            "total_orders": self.total_orders_processed,
            "total_volume": self.total_volume_matched,
            "total_fills": self.fill_count
        }

    def drain_fills(self, uint64_t start_idx=0):
        """Returns new execution fills as Python tuples for ledger audit logging."""
        fills = []
        cdef uint64_t i
        for i in range(start_idx, self.fill_count):
            fills.append((
                self.fill_buffer[i].match_id,
                self.fill_buffer[i].buy_order_id,
                self.fill_buffer[i].sell_order_id,
                self.fill_buffer[i].fill_price,
                self.fill_buffer[i].fill_qty,
                self.fill_buffer[i].engine_latency_ns
            ))
        return fills

    def get_latency_stats(self):
        """Computes statistical percentiles (p50, p90, p99, p99.9) in microseconds."""
        if self.latency_count == 0:
            return {"count": 0, "p50_us": 0.0, "p90_us": 0.0, "p99_us": 0.0, "p99_9_us": 0.0, "max_us": 0.0, "mean_us": 0.0}

        # Convert nanoseconds to microseconds
        cdef list samples = [self.latency_records_ns[i] / 1000.0 for i in range(self.latency_count)]
        samples.sort()

        cdef size_t n = len(samples)
        cdef double p50 = samples[int(n * 0.50)]
        cdef double p90 = samples[int(n * 0.90)]
        cdef double p99 = samples[int(n * 0.99)]
        cdef double p99_9 = samples[int(n * 0.999)]
        cdef double max_lat = samples[-1]
        cdef double mean_lat = sum(samples) / n

        return {
            "count": n,
            "p50_us": p50,
            "p90_us": p90,
            "p99_us": p99,
            "p99_9_us": p99_9,
            "max_us": max_lat,
            "mean_us": mean_lat
        }
