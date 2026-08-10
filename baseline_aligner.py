import time
import os
import psutil
from typing import List, Tuple

# Configuration / Hyperparameters
GENOME_SIZE_SIMULATION = 5_000_000  # 5 Million BP for quick validation (Scale up for full genome benchmarks)
TARGET_SGRNA = "GAGTCCGAGCAGAAGAAGAA"  # 20-bp guide RNA
PAM_PATTERN = "AGG"                     # 3-bp PAM site (NGG)
QUERY_SEQ = TARGET_SGRNA + PAM_PATTERN  # Total length M = 23
MAX_MISMATCHES = 4                     # CRISPR mismatch tolerance threshold

def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# -----------------------------------------------------------------
# 1. Naive Brute-Force Off-Target Matcher O(N * M)
# -----------------------------------------------------------------
def run_brute_force_matcher(genome: str, query: str, max_mismatches: int) -> List[Tuple[int, int]]:
    """
    Linearly steps through the genome text character-by-character to find 
    off-target matches within the mismatch tolerance.
    
    Returns: List of tuples containing (genomic_index, mismatch_count)
    """
    n = len(genome)
    m = len(query)
    hits = []

    for i in range(n - m + 1):
        mismatches = 0
        
        # Check sgRNA region (first 20 bp)
        for j in range(m - 3):
            if genome[i + j] != query[j]:
                mismatches += 1
                if mismatches > max_mismatches:
                    break
        
        if mismatches <= max_mismatches:
            # Verify NGG PAM site (positions 21 and 22 in 0-indexed window)
            if genome[i + 21] == 'G' and genome[i + 22] == 'G':
                hits.append((i, mismatches))
                
    return hits

# -------------------------------------------------------------------------
# 2. Smith-Waterman Local Alignment Algorithm O(N * M)
# -------------------------------------------------------------------------
def run_smith_waterman_scores(
    genome: str, 
    query: str, 
    match_score: int = 2, 
    mismatch_penalty: int = -1, 
    gap_penalty: int = -2
) -> List[Tuple[int, int]]:
    """
    Space-optimized Smith-Waterman local alignment scoring.
    Uses 2 rows (O(M) space) to compute alignment scores across sequence N.
    """
    n = len(genome)
    m = len(query)
    
    prev_row = [0] * (m + 1)
    curr_row = [0] * (m + 1)
    
    high_scoring_sites = []
    threshold = (m - MAX_MISMATCHES) * match_score + (MAX_MISMATCHES * mismatch_penalty)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if genome[i - 1] == query[j - 1]:
                score = match_score
            else:
                score = mismatch_penalty
                
            match = prev_row[j - 1] + score
            delete = prev_row[j] + gap_penalty
            insert = curr_row[j - 1] + gap_penalty
            
            curr_row[j] = max(0, match, delete, insert)
            
            if j == m and curr_row[j] >= threshold:
                high_scoring_sites.append((i - m, curr_row[j]))
                
        prev_row = list(curr_row)
        curr_row = [0] * (m + 1)
        
    return high_scoring_sites

# -------------------------------------------------------------------------
# Synthetic Data Generator & Benchmark Execution
# -------------------------------------------------------------------------
def generate_synthetic_dna(length: int) -> str:
    """Generates a pseudo-random DNA string composed of {A, C, G, T}."""
    import random
    random.seed(42)  # Deterministic seed for reproducible benchmarks
    return "".join(random.choices(["A", "C", "G", "T"], k=length))

if __name__ == "__main__":
    print("=" * 65)
    print("      GeneWeaver: CPU Alignment Baseline Benchmark Suite      ")
    print("=" * 65)
    
    print(f"Generating synthetic genomic data ({GENOME_SIZE_SIMULATION:,} bp)...")
    mem_before_gen = get_process_memory_mb()
    genome_seq = generate_synthetic_dna(GENOME_SIZE_SIMULATION)
    
    # Inject a known off-target match at position 1,000,000
    inject_pos = 1_000_000
    mutated_target = list("GAGTCCGAGCAGAAGAACAA" + "AGG")  # 1 mismatch (G -> C at pos 17)
    genome_seq = genome_seq[:inject_pos] + "".join(mutated_target) + genome_seq[inject_pos+23:]
    
    print(f"Memory Overhead for Sequence: {get_process_memory_mb() - mem_before_gen:.2f} MB")
    print(f"Target sgRNA + PAM: {QUERY_SEQ} (Length: {len(QUERY_SEQ)} bp)")
    print("-" * 65)

    # Benchmark 1: Brute-Force Matcher
    print("\n[Benchmark 1] Executing Naive Brute-Force Matcher...")
    start_time = time.perf_counter()
    start_mem = get_process_memory_mb()
    
    bf_hits = run_brute_force_matcher(genome_seq, QUERY_SEQ, MAX_MISMATCHES)
    
    bf_duration = time.perf_counter() - start_time
    bf_mem = get_process_memory_mb() - start_mem
    
    print(f"  └── Time Elapsed:    {bf_duration:.4f} seconds")
    print(f"  └── Off-Target Hits: {len(bf_hits)} found")
    print(f"  └── Processing Rate: {(GENOME_SIZE_SIMULATION / bf_duration) / 1e6:.3f} Million BP/sec")
    
    # Benchmark 2: Smith-Waterman Local Alignment
    print("\n[Benchmark 2] Executing Space-Optimized Smith-Waterman...")
    start_time = time.perf_counter()
    start_mem = get_process_memory_mb()
    
    sw_hits = run_smith_waterman_scores(genome_seq, QUERY_SEQ)
    
    sw_duration = time.perf_counter() - start_time
    sw_mem = get_process_memory_mb() - start_mem
    
    print(f"  └── Time Elapsed:    {sw_duration:.4f} seconds")
    print(f"  └── High-Score Hits: {len(sw_hits)} found")
    print(f"  └── Processing Rate: {(GENOME_SIZE_SIMULATION / sw_duration) / 1e6:.3f} Million BP/sec")

    # Extrapolate to 3.2 Gigabase Human Genome
    extrapolated_bf_hours = (bf_duration / GENOME_SIZE_SIMULATION * 3.2e9) / 3600
    extrapolated_sw_hours = (sw_duration / GENOME_SIZE_SIMULATION * 3.2e9) / 3600
    
    print("\n" + "=" * 65)
    print("          PROJECTED 3.2 Gb HUMAN GENOME BASELINE               ")
    print("=" * 65)
    print(f"  • Brute-Force Matcher Estimated Time: {extrapolated_bf_hours:.2f} Hours")
    print(f"  • Smith-Waterman Estimated Time:     {extrapolated_sw_hours:.2f} Hours")
    print("=" * 65)