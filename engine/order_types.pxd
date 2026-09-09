# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# ==============================================================================
# ChronosMatch: Zero-Copy High-Frequency Trading Engine
# Module: engine/order_types.pxd
# Role: Person 1 (Low-Latency Core & Memory Architect)
# Description: Packed C-level struct definitions matching the 32-byte shared
#              memory binary protocol contract and cache-aligned ring buffer headers.
# ==============================================================================

from libc.stdint cimport uint64_t, uint32_t, uint16_t, uint8_t, int32_t

cdef extern from *:
    """
    #pragma pack(push, 1)
    typedef struct {
        uint64_t order_id;      /* 8 bytes: Offset 0  */
        uint64_t timestamp_ns;  /* 8 bytes: Offset 8  */
        double   price;         /* 8 bytes: Offset 16 */
        uint32_t qty;           /* 4 bytes: Offset 24 */
        char     side;          /* 1 byte : Offset 28 ('B'=Buy, 'S'=Sell) */
        char     order_type;    /* 1 byte : Offset 29 ('L'=Limit, 'M'=Market, 'C'=Cancel) */
        uint16_t padding;       /* 2 bytes: Offset 30 (Aligment to 32 bytes) */
    } C_OrderTick;

    typedef struct {
        uint64_t match_id;          /* 8 bytes: Unique execution sequence ID */
        uint64_t buy_order_id;      /* 8 bytes: Matched buy order */
        uint64_t sell_order_id;     /* 8 bytes: Matched sell order */
        double   fill_price;        /* 8 bytes: Execution price */
        uint32_t fill_qty;          /* 4 bytes: Filled volume */
        uint32_t padding;           /* 4 bytes: Alignment padding */
        uint64_t engine_latency_ns; /* 8 bytes: Ingestion-to-Fill latency (nanoseconds) */
    } C_ExecutionFill;

    typedef struct {
        /* Cache Line 1: Producer Write Line */
        volatile uint64_t head;     /* 8 bytes: Producer write index */
        uint8_t  pad1[56];          /* 56 bytes padding to 64B cache boundary */

        /* Cache Line 2: Consumer Read Line */
        volatile uint64_t tail;     /* 8 bytes: Consumer read index */
        uint8_t  pad2[56];          /* 56 bytes padding to 64B cache boundary */

        /* Cache Line 3: Static Geometry & Metadata */
        uint64_t capacity;          /* 8 bytes: Total ring slots (power of 2) */
        uint64_t mask;              /* 8 bytes: Bitmask (capacity - 1) */
        uint8_t  pad3[48];          /* 48 bytes padding to 64B cache boundary */
    } C_RingBufferHeader;
    #pragma pack(pop)
    """
    ctypedef struct C_OrderTick:
        uint64_t order_id
        uint64_t timestamp_ns
        double   price
        uint32_t qty
        char     side
        char     order_type
        uint16_t padding

    ctypedef struct C_ExecutionFill:
        uint64_t match_id
        uint64_t buy_order_id
        uint64_t sell_order_id
        double   fill_price
        uint32_t fill_qty
        uint32_t padding
        uint64_t engine_latency_ns

    ctypedef struct C_RingBufferHeader:
        uint64_t head
        uint8_t  pad1[56]
        uint64_t tail
        uint8_t  pad2[56]
        uint64_t capacity
        uint64_t mask
        uint8_t  pad3[48]

# Cython type aliases
ctypedef C_OrderTick OrderTick
ctypedef C_ExecutionFill ExecutionFill
ctypedef C_RingBufferHeader RingBufferHeader

# Internal Order Book Node for pre-allocated slab allocator (Zero-GC)
cdef struct OrderNode:
    uint64_t order_id
    uint64_t timestamp_ns
    double   price
    uint32_t qty
    char     side
    char     order_type
    int32_t  prev_idx       # Doubly-linked list pointer (slab index, -1 if none)
    int32_t  next_idx       # Doubly-linked list pointer (slab index, -1 if none)
    int32_t  price_level_idx

# Price Level Aggregator Node
cdef struct PriceLevel:
    double   price
    uint32_t total_volume
    uint32_t order_count
    int32_t  head_idx       # Index of oldest order in FIFO queue
    int32_t  tail_idx       # Index of newest order in FIFO queue
    int32_t  prev_level     # Linked list of sorted price levels (-1 if none)
    int32_t  next_level
