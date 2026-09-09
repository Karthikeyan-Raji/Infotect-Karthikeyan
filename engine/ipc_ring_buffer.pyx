# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# ==============================================================================
# ChronosMatch: Zero-Copy High-Frequency Trading Engine
# Module: engine/ipc_ring_buffer.pyx
# Role: Person 1 (Low-Latency Core & Memory Architect)
# Description: Lockless SPSC (Single-Producer Single-Consumer) Shared Memory Ring
#              Buffer with cache-line padding, atomic synchronization, and zero-copy mmap.
# ==============================================================================

from libc.stdint cimport uint64_t, uint32_t, uint8_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from engine.order_types cimport OrderTick, ExecutionFill, RingBufferHeader

import os
import mmap
import struct
import platform

cdef extern from *:
    """
    #if defined(_MSC_VER)
        #include <windows.h>
        #include <intrin.h>
        #define MEMORY_BARRIER() MemoryBarrier()
        #define COMPILER_BARRIER() _ReadWriteBarrier()
        static inline uint64_t atomic_load_u64_acquire(volatile uint64_t* ptr) {
            uint64_t val = *ptr;
            MemoryBarrier();
            return val;
        }
        static inline void atomic_store_u64_release(volatile uint64_t* ptr, uint64_t val) {
            MemoryBarrier();
            *ptr = val;
        }
    #else
        #define MEMORY_BARRIER() __sync_synchronize()
        #define COMPILER_BARRIER() __asm__ __volatile__("" ::: "memory")
        static inline uint64_t atomic_load_u64_acquire(volatile uint64_t* ptr) {
            return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
        }
        static inline void atomic_store_u64_release(volatile uint64_t* ptr, uint64_t val) {
            __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
        }
    #endif
    """
    void MEMORY_BARRIER() nogil
    void COMPILER_BARRIER() nogil
    uint64_t atomic_load_u64_acquire(volatile uint64_t* ptr) nogil
    uint64_t atomic_store_u64_release(volatile uint64_t* ptr, uint64_t val) nogil


cdef class SPSCBufferReader:
    """
    Zero-copy consumer reading directly from mapped POSIX shared memory (/dev/shm).
    Runs with nogil in the high-frequency matching loop.
    """
    cdef:
        int fd
        size_t total_size
        char* base_ptr
        RingBufferHeader* header
        OrderTick* slots
        uint64_t capacity
        uint64_t mask
        object mmap_obj
        str file_path

    def __cinit__(self, str file_path):
        self.file_path = file_path
        self.fd = -1
        self.base_ptr = NULL
        self.header = NULL
        self.slots = NULL
        self.capacity = 0
        self.mask = 0
        self.mmap_obj = None

    def map_buffer(self):
        """Map the existing shared memory segment created by initialization harness."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Shared memory file not found: {self.file_path}")

        file_size = os.path.getsize(self.file_path)
        self.total_size = file_size

        f = open(self.file_path, "r+b")
        self.mmap_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
        f.close()

        # Extract underlying C pointer from mmap object buffer protocol
        cdef const void* buf_ptr
        cdef Py_ssize_t buf_len
        # Using buffer interface
        self.base_ptr = <char*><void*><uintptr_t>(id(self.mmap_obj)) # fallback dummy
        # Cython allows direct address acquisition from memoryview / buffer
        cdef unsigned char[:] mem_view = self.mmap_obj
        self.base_ptr = <char*>&mem_view[0]

        self.header = <RingBufferHeader*>self.base_ptr
        self.capacity = self.header.capacity
        self.mask = self.header.mask
        self.slots = <OrderTick*>(self.base_ptr + sizeof(RingBufferHeader))

    cdef inline bint poll_next(self, OrderTick* out_tick) nogil:
        """
        Polls for next available tick in ring buffer.
        Returns 1 (True) if tick was dequeued, 0 (False) if empty.
        Zero Python GIL involvement, pure memory barriers.
        """
        cdef uint64_t current_tail = self.header.tail
        cdef uint64_t current_head = atomic_load_u64_acquire(&self.header.head)

        if current_tail >= current_head:
            return 0  # Buffer empty

        # Zero-copy slot read
        cdef uint64_t slot_idx = current_tail & self.mask
        out_tick[0] = self.slots[slot_idx]

        # Release index update
        atomic_store_u64_release(&self.header.tail, current_tail + 1)
        return 1

    def read_one(self):
        """Python-accessible method for testing single tick consumption."""
        cdef OrderTick tick
        cdef bint has_tick
        with nogil:
            has_tick = self.poll_next(&tick)

        if not has_tick:
            return None

        return {
            "order_id": tick.order_id,
            "timestamp_ns": tick.timestamp_ns,
            "price": tick.price,
            "qty": tick.qty,
            "side": chr(tick.side),
            "order_type": chr(tick.order_type),
        }

    def close(self):
        if self.mmap_obj is not None:
            self.mmap_obj.close()
            self.mmap_obj = None
            self.base_ptr = NULL


cdef class SPSCBufferWriter:
    """
    High-throughput ring buffer writer used by the ingestion process or market simulator.
    Supports atomic pointer math, power-of-two modulo masking, and cache line isolation.
    """
    cdef:
        int fd
        size_t total_size
        char* base_ptr
        RingBufferHeader* header
        OrderTick* slots
        uint64_t capacity
        uint64_t mask
        object mmap_obj
        str file_path

    def __cinit__(self, str file_path, uint64_t capacity=1048576):
        # Default capacity: 2^20 (1,048,576 orders)
        self.file_path = file_path
        self.capacity = capacity
        self.mask = capacity - 1
        self.total_size = sizeof(RingBufferHeader) + (capacity * sizeof(OrderTick))
        self.mmap_obj = None
        self.base_ptr = NULL

    def init_and_map(self):
        """Create or truncate shared memory file and write initial ring buffer header."""
        # Ensure parent directory exists
        parent_dir = os.path.dirname(self.file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(self.file_path, "wb") as f:
            f.seek(self.total_size - 1)
            f.write(b"\0")
            f.flush()

        f = open(self.file_path, "r+b")
        self.mmap_obj = mmap.mmap(f.fileno(), self.total_size, access=mmap.ACCESS_WRITE)
        f.close()

        cdef unsigned char[:] mem_view = self.mmap_obj
        self.base_ptr = <char*>&mem_view[0]
        memset(self.base_ptr, 0, self.total_size)

        self.header = <RingBufferHeader*>self.base_ptr
        self.header.head = 0
        self.header.tail = 0
        self.header.capacity = self.capacity
        self.header.mask = self.mask
        self.slots = <OrderTick*>(self.base_ptr + sizeof(RingBufferHeader))

    cdef inline bint push_tick(self, OrderTick* tick) nogil:
        """
        Appends an order tick to the ring buffer.
        Returns 1 on success, 0 if buffer is full (backpressure condition).
        """
        cdef uint64_t current_head = self.header.head
        cdef uint64_t current_tail = atomic_load_u64_acquire(&self.header.tail)

        # Check full: head - tail >= capacity
        if (current_head - current_tail) >= self.capacity:
            return 0  # Ring buffer full

        cdef uint64_t slot_idx = current_head & self.mask
        self.slots[slot_idx] = tick[0]

        # Publish slot via release store
        atomic_store_u64_release(&self.header.head, current_head + 1)
        return 1

    def write_order(self, uint64_t order_id, uint64_t timestamp_ns, double price,
                    uint32_t qty, str side, str order_type):
        """Python entry point to enqueue an order."""
        cdef OrderTick tick
        tick.order_id = order_id
        tick.timestamp_ns = timestamp_ns
        tick.price = price
        tick.qty = qty
        tick.side = ord(side[0])
        tick.order_type = ord(order_type[0])
        tick.padding = 0

        cdef bint success
        with nogil:
            success = self.push_tick(&tick)
        return bool(success)

    def close(self):
        if self.mmap_obj is not None:
            self.mmap_obj.close()
            self.mmap_obj = None
            self.base_ptr = NULL
