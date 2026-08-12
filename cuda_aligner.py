import numpy as np
import time
import os
import psutil

try:
    from numba import cuda
    _NUMBA_CUDA_IMPORT_ERROR = None
except Exception as err:
    cuda = None
    _NUMBA_CUDA_IMPORT_ERROR = err

# =========================================================================
# Hyperparameters & Constants
# =========================================================================
THREADS_PER_BLOCK = 256
MAX_HITS_CAPACITY = 100_000  # Device output array buffer size

# 2-Bit Nucleotide Encoding Map: A=00, C=01, G=10, T=11
NUC_BIT_MAP = np.full(256, 0, dtype=np.uint8)
NUC_BIT_MAP[ord('A')] = 0b00
NUC_BIT_MAP[ord('C')] = 0b01
NUC_BIT_MAP[ord('G')] = 0b10
NUC_BIT_MAP[ord('T')] = 0b11

BIT_TO_NUC = {0b00: 'A', 0b01: 'C', 0b10: 'G', 0b11: 'T'}


def cuda_available() -> bool:
    if cuda is None:
        return False
    try:
        return cuda.is_available()
    except Exception:
        return False


def cuda_error_message() -> str:
    if cuda is None:
        if _NUMBA_CUDA_IMPORT_ERROR is not None:
            return str(_NUMBA_CUDA_IMPORT_ERROR)
        return 'Numba CUDA could not be imported.'
    try:
        if not cuda.is_available():
            return 'CUDA is installed but not available on this machine.'
    except Exception as err:
        return str(err)
    return 'Unknown CUDA error.'


def pack_dna_sequence(dna_str: str) -> np.ndarray:
    ascii_bytes = np.frombuffer(dna_str.upper().encode('ascii'), dtype=np.uint8)
    mapped_2bit = NUC_BIT_MAP[ascii_bytes]
    pad_len = (4 - (len(mapped_2bit) % 4)) % 4
    if pad_len > 0:
        mapped_2bit = np.pad(mapped_2bit, (0, pad_len), 'constant', constant_values=0)
    packed = (mapped_2bit[0::4] << 6) |              (mapped_2bit[1::4] << 4) |              (mapped_2bit[2::4] << 2) |              (mapped_2bit[3::4])
    return packed.astype(np.uint8)


def unpack_2bit_window(packed_arr: np.ndarray, byte_idx: int, nucleotide_len: int = 23) -> str:
    chars = []
    for i in range(nucleotide_len):
        b_idx = byte_idx + (i // 4)
        shift = 6 - 2 * (i % 4)
        bit_val = (packed_arr[b_idx] >> shift) & 0b11
        chars.append(BIT_TO_NUC[bit_val])
    return ''.join(chars)


def run_cpu_fallback(genome_str: str, target_sgrna_str: str, max_mismatch: int = 4):
    print('[WARNING] CUDA unavailable, using CPU fallback.')
    from baseline_aligner import run_brute_force_matcher
    query = target_sgrna_str + 'AGG'
    hits = run_brute_force_matcher(genome_str, query, max_mismatch)
    return {
        'hits_count': len(hits),
        'indices': np.array([pos for pos, _ in hits], dtype=np.int32),
        'mismatches': np.array([mm for _, mm in hits], dtype=np.int32),
        'h2d_time': 0.0,
        'kernel_time': 0.0,
        'd2h_time': 0.0,
        'total_gpu_time': 0.0,
    }


if cuda is not None:
    @cuda.jit
    def crispr_offtarget_kernel(
        d_genomic_packed,
        packed_len,
        genome_bp_len,
        target_encoded,
        max_mismatches,
        d_out_indices,
        d_out_mismatches,
        d_hit_count,
    ):
        global_bp_idx = cuda.grid(1)
        if global_bp_idx + 23 > genome_bp_len:
            return

        byte_offset = global_bp_idx // 4
        bit_shift_offset = (global_bp_idx % 4) * 2
        if byte_offset + 6 >= packed_len:
            return

        raw_64 = np.uint64(0)
        for b in range(7):
            raw_64 = (raw_64 << np.uint64(8)) | np.uint64(d_genomic_packed[byte_offset + b])

        shifted_window = raw_64 << np.uint64(bit_shift_offset)
        sgrna_window = np.uint64(shifted_window >> np.uint64(24))
        target_sgrna = np.uint64(target_encoded)

        xor_diff = sgrna_window ^ target_sgrna
        mismatches = 0
        for i in range(20):
            pair_diff = (xor_diff >> np.uint64(38 - 2 * i)) & np.uint64(0b11)
            if pair_diff != np.uint64(0):
                mismatches += 1
                if mismatches > max_mismatches:
                    return

        pam_bp1 = (shifted_window >> np.uint64(22)) & np.uint64(0b11)
        pam_bp2 = (shifted_window >> np.uint64(20)) & np.uint64(0b11)
        if pam_bp1 == np.uint64(0b10) and pam_bp2 == np.uint64(0b10):
            hit_slot = cuda.atomic.add(d_hit_count, 0, 1)
            if hit_slot < d_out_indices.size:
                d_out_indices[hit_slot] = global_bp_idx
                d_out_mismatches[hit_slot] = mismatches
else:
    def crispr_offtarget_kernel(*args, **kwargs):
        raise RuntimeError('CUDA kernel is unavailable because Numba CUDA could not be imported.')


def run_gpu_alignment(genome_str: str, target_sgrna_str: str, max_mismatch: int = 4):
    if len(genome_str) < 23:
        return {
            'hits_count': 0,
            'indices': np.array([], dtype=np.int32),
            'mismatches': np.array([], dtype=np.int32),
            'h2d_time': 0.0,
            'kernel_time': 0.0,
            'd2h_time': 0.0,
            'total_gpu_time': 0.0,
        }

    if not cuda_available():
        print(f'[WARNING] {cuda_error_message()}')
        return run_cpu_fallback(genome_str, target_sgrna_str, max_mismatch)

    print(f'1. Packing genomic string ({len(genome_str):,} bp) into 2-bit representation...')
    packed_genome = pack_dna_sequence(genome_str)
    packed_len = len(packed_genome)
    genome_bp_len = len(genome_str)

    packed_target = pack_dna_sequence(target_sgrna_str)
    if len(packed_target) < 5:
        raise ValueError('Target sgRNA string must be at least 20 bases long.')

    target_64 = np.uint64(0)
    for b in range(5):
        target_64 = (target_64 << np.uint64(8)) | np.uint64(packed_target[b])

    print('2. Transferring data from Host RAM to Device VRAM (H2D)...')
    h2d_start = time.perf_counter()

    d_genomic_packed = cuda.to_device(packed_genome)
    d_out_indices = cuda.device_array(MAX_HITS_CAPACITY, dtype=np.int32)
    d_out_mismatches = cuda.device_array(MAX_HITS_CAPACITY, dtype=np.int32)
    d_hit_count = cuda.to_device(np.zeros(1, dtype=np.int32))

    cuda.synchronize()
    h2d_time = time.perf_counter() - h2d_start

    total_threads = genome_bp_len - 23 + 1
    blocks_per_grid = (total_threads + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    print(f'3. Launching CUDA Kernel across {blocks_per_grid:,} thread blocks ({THREADS_PER_BLOCK} threads/block)...')
    kernel_start = time.perf_counter()
    try:
        crispr_offtarget_kernel[blocks_per_grid, THREADS_PER_BLOCK](
            d_genomic_packed,
            packed_len,
            genome_bp_len,
            target_64,
            max_mismatch,
            d_out_indices,
            d_out_mismatches,
            d_hit_count,
        )
        cuda.synchronize()
    except Exception as err:
        print(f'[WARNING] CUDA kernel launch failed: {err}')
        return run_cpu_fallback(genome_str, target_sgrna_str, max_mismatch)

    kernel_time = time.perf_counter() - kernel_start

    print('4. Transferring results from Device VRAM back to Host RAM (D2H)...')
    d2h_start = time.perf_counter()

    total_hits = d_hit_count.copy_to_host()[0]
    hits_indices = d_out_indices[:total_hits].copy_to_host()
    hits_mismatches = d_out_mismatches[:total_hits].copy_to_host()
    d2h_time = time.perf_counter() - d2h_start

    return {
        'hits_count': total_hits,
        'indices': hits_indices,
        'mismatches': hits_mismatches,
        'h2d_time': h2d_time,
        'kernel_time': kernel_time,
        'd2h_time': d2h_time,
        'total_gpu_time': h2d_time + kernel_time + d2h_time,
    }


if __name__ == '__main__':
    print('=' * 65)
    print('        GeneWeaver: Functional Numba CUDA Alignment Kernel        ')
    print('=' * 65)

    GENOME_LEN = 5_000_000
    TARGET_SGRNA = 'GAGTCCGAGCAGAAGAAGAA'
    PAM_SITE = 'AGG'

    print(f'Generating synthetic DNA sequence ({GENOME_LEN:,} bp)...')
    np.random.seed(42)
    bases = np.array(['A', 'C', 'G', 'T'])
    random_seq = np.random.choice(bases, size=GENOME_LEN)
    genome_dna = ''.join(random_seq)

    pos1 = 500_000
    genome_dna = genome_dna[:pos1] + (TARGET_SGRNA + PAM_SITE) + genome_dna[pos1 + 23:]

    pos2 = 2_500_000
    mutated_sgRNA = 'GCGTCCGACTAGAAGAAGAA'
    genome_dna = genome_dna[:pos2] + (mutated_sgRNA + PAM_SITE) + genome_dna[pos2 + 23:]

    results = run_gpu_alignment(genome_dna, TARGET_SGRNA, max_mismatch=4)

    print('' + '=' * 65)
    print('                      KERNEL RUN SUMMARY                       ')
    print('=' * 65)
    print(f'Total Off-Target Hits Found: {results['hits_count']}')
    print(f'  ├── Host-to-Device (H2D) Transfer Time: {results['h2d_time']:.4f} s')
    print(f'  ├── Pure Kernel Execution Time:       {results['kernel_time']:.4f} s')
    print(f'  ├── Device-to-Host (D2H) Transfer Time: {results['d2h_time']:.4f} s')
    print(f'  └── Total GPU Stage Runtime:          {results['total_gpu_time']:.4f} s')
    print('-' * 65)

    print('Verifying Accuracy of Detected Off-Targets:')
    for idx, mismatches in zip(results['indices'], results['mismatches']):
        extracted = genome_dna[idx : idx + 23]
        print(f'  • Position {idx:10,d} | Mismatches: {mismatches} | Found: {extracted}')

    assert pos1 in results['indices'], f'Error: Failed to find injected target at {pos1}'
    assert pos2 in results['indices'], f'Error: Failed to find injected target at {pos2}'
    print('[SUCCESS] Correctness check passed! All injected targets accurately detected.')
