import os, io, json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from multiprocessing import Pool, cpu_count
import zstandard as zstd
from functools import partial


TRAIN_TOKENS = 50_000_000
# TRAIN_TOKENS = 50_000_000

VALIDATION_DOCUMENTS = 50_000
# VALIDATION_DOCUMENTS = 150_000


def _zstd_size(text, level=3):
    """Compress a string using zstd and return the compressed size"""
    c = zstd.ZstdCompressor(level=level)
    original_size = len(text.encode('utf-8'))
    compressed = c.compress(text.encode('utf-8'))
    size = len(compressed)
    print(f"  Zstd compression: {original_size:,} bytes → {size:,} bytes (ratio: {size/original_size:.3f})")
    return size

def _zstd_concat_size(texts, level=3):
    """Concatenate strings and compress the result"""
    c = zstd.ZstdCompressor(level=level)
    concatenated = "".join(texts)
    original_size = len(concatenated.encode('utf-8'))
    compressed = c.compress(concatenated.encode('utf-8'))
    size = len(compressed)
    print(f"  Zstd concatenation: {original_size:,} bytes → {size:,} bytes (ratio: {size/original_size:.3f})")
    return size

def get_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def build_strings_from_adapt(hf_path):
    ds = load_dataset(hf_path, streaming=True)
    strings = []
    for split in ds:
        i = 0
        for ex in tqdm(ds[split], desc=f"Building {hf_path}"):
            if i > VALIDATION_DOCUMENTS:
                return strings
            msgs = ex.get("messages", [])
            parts = []
            for m in msgs:
                if "content" in m:
                    parts.append(m["content"])
            if parts:
                strings.append("".join(parts))
            i += 1
    return strings

def detokenize_chunk(chunk_data):
    """Detokenize a chunk of tokens using the provided tokenizer"""
    chunk_tokens, tokenizer_name = chunk_data
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    # Process the chunk in smaller batches to avoid memory issues
    batch_size = 1024
    texts = []
    
    for i in range(0, len(chunk_tokens), batch_size):
        batch_tokens = chunk_tokens[i:i+batch_size]
        text = tok.decode(batch_tokens, skip_special_tokens=True)
        texts.append(text)
    
    return "".join(texts)

def build_strings_from_tokens(token_file_path, max_tokens=None):
    """Build strings by detokenizing from a token file"""
    tokenizer_name = "allenai/dolma2-tokenizer"
    
    # Load tokens
    tokens = np.memmap(token_file_path, mode="r", dtype=np.int32)
    if max_tokens:
        tokens = tokens[:max_tokens]
    
    # Split into chunks for processing
    chunk_size = 100_000  # Process in chunks to avoid memory issues
    chunks = []
    
    for chunk_start in range(0, len(tokens), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(tokens))
        chunk_tokens = tokens[chunk_start:chunk_end]
        chunks.append((chunk_tokens, tokenizer_name))
    
    # Process chunks sequentially (multiprocessing will be handled at main level)
    chunk_texts = []
    for chunk in tqdm(chunks, desc="Processing token chunks"):
        chunk_texts.append(detokenize_chunk(chunk))
    
    # Join all chunk texts into one string
    return "".join(chunk_texts)

def process_subset_with_text(args):
    """Process subset using pre-processed text"""
    subset, train_text, val_text, cx_val, level = args
    
    print(f"\nProcessing subset: {subset}")
    print(f"  Train text size: {len(train_text.encode('utf-8')):,} bytes")
    print(f"  Val text size: {len(val_text.encode('utf-8')):,} bytes")
    print(f"  Val compressed size (cx_val): {cx_val:,} bytes")
    
    cy = _zstd_size(train_text, level)
    
    # Calculate compressed sizes of concatenations
    cxy = _zstd_concat_size([val_text, train_text], level)
    cyx = _zstd_concat_size([train_text, val_text], level)
    
    # Calculate mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
    mi1 = cx_val + cy - cxy
    mi2 = cx_val + cy - cyx
    mi = (mi1 + mi2) // 2
    
    print(f"  MI calculation: {cx_val:,} + {cy:,} - {cxy:,} = {mi1:,}")
    print(f"  MI calculation: {cx_val:,} + {cy:,} - {cyx:,} = {mi2:,}")
    print(f"  Final MI: {mi:,} bytes")
    
    return subset, len(train_text.encode('utf-8')), mi

def process_token_file(args):
    """Process a single token file to extract text"""
    base_path, subset, train_cap = args
    data_path = f"{base_path}/{subset}/dolma2-tokenizer/part-000-00000.npy"
    
    # Build string from tokens
    train_text = build_strings_from_tokens(data_path, train_cap)
    
    return subset, train_text

def main():

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    base_path = "/oe-training-default/ai2-llm/preprocessed/dclm/baseline_topic_classified_sample"
    level = int(os.environ.get("ZSTD_LEVEL", "3"))
    strings = build_strings_from_adapt("allenai/tulu-3-sft-mixture")
    val_text = "".join(strings)  # Concatenate all validation strings
    cx_val = _zstd_size(val_text, level)
    console = Console()
    table = Table(title="mutual information via Zstd compression of raw text (symmetric)")
    table.add_column("Subset", style="cyan", no_wrap=True)
    table.add_column("Train MB", style="magenta", justify="right")
    table.add_column("Val MB", style="magenta", justify="right")
    table.add_column("Mutual Information", style="green", justify="right")
    subsets = get_dirs(base_path)
    
    # Use all available CPUs for processing
    num_processes = cpu_count()
    
    # First, process all token files in parallel to extract text
    print("Processing token files to extract text...")
    token_tasks = [(base_path, s, TRAIN_TOKENS) for s in subsets]
    
    subset_texts = {}
    with Pool(processes=num_processes) as pool:
        for subset, train_text in tqdm(pool.imap_unordered(process_token_file, token_tasks), total=len(token_tasks), desc="Processing token files"):
            subset_texts[subset] = train_text
    
    # Now process mutual information calculations in parallel using pre-processed text
    print("Calculating mutual information...")
    mi_tasks = [(s, subset_texts[s], val_text, cx_val, level) for s in subsets]
    
    results = []
    with Pool(processes=num_processes) as pool:
        for subset, ntrain_bytes, mi in tqdm(pool.imap_unordered(process_subset_with_text, mi_tasks), total=len(mi_tasks), desc="Processing subsets"):
            results.append((subset, ntrain_bytes, mi))
    
    # Sort results by mutual information (descending - highest MI first)
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Add sorted results to table
    for subset, ntrain_bytes, mi in results:
        train_mb = ntrain_bytes / (1024 * 1024)
        val_mb = len(val_text.encode('utf-8')) / (1024 * 1024)
        table.add_row(subset, f"{train_mb:.3f}", f"{val_mb:.3f}", f"{mi:,} bytes")
    
    console.print(table)

if __name__ == "__main__":
    main()