import os, io, json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from multiprocessing import Pool, cpu_count
import zstandard as zstd

def _zstd_size(arr, level=3):
    c = zstd.ZstdCompressor(level=level)
    a = np.ascontiguousarray(arr, dtype=np.int32)
    return len(c.compress(memoryview(a)))

def _zstd_stream_concat_size(arrs, level=3):
    c = zstd.ZstdCompressor(level=level)
    out = io.BytesIO()
    with c.stream_writer(out) as w:
        for a in arrs:
            b = memoryview(np.ascontiguousarray(a, dtype=np.int32))
            w.write(b)
        # Get the position before the context manager exits
        w.flush()
        result = out.tell()
    return result

def get_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def build_strings_from_adapt(hf_path):
    ds = load_dataset(hf_path, streaming=True)
    strings = []
    for split in ds:
        i = 0
        for ex in tqdm(ds[split], desc=f"Building {hf_path}"):
            if i > 10_000:
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

def build_tokens(strings):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer", use_fast=True)
    toks = []
    bs = 1024
    for i in tqdm(range(0, len(strings), bs), desc="Tokenizing"):
        batch = strings[i:i+bs]
        enc = tok(batch, add_special_tokens=False).encodings
        for e in enc:
            toks.extend(e.ids)
    return np.asarray(toks, dtype=np.int32)

def process_subset(args):
    base_path, subset, val_toks, cx_val, train_cap, level = args
    data_path = f"{base_path}/{subset}/dolma2-tokenizer/part-000-00000.npy"
    train_toks = np.memmap(data_path, mode="r", dtype=np.int32)[:train_cap]
    cy = _zstd_size(train_toks, level)
    cxy = _zstd_stream_concat_size([val_toks, train_toks], level)
    cyx = _zstd_stream_concat_size([train_toks, val_toks], level)
    
    # Calculate mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
    # where H(X) = cx_val, H(Y) = cy, H(X,Y) = cxy or cyx
    mi1 = cx_val + cy - cxy
    mi2 = cx_val + cy - cyx
    mi = (mi1 + mi2) // 2
    
    # # Mutual information cannot be negative (represents shared information)
    # # If negative, it means concatenation actually increases compressed size
    # # due to lack of shared patterns between sequences
    # if mi < 0:
    #     mi = 0
    
    return subset, len(train_toks), mi

def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    base_path = "/oe-training-default/ai2-llm/preprocessed/dclm/baseline_topic_classified_sample"
    level = int(os.environ.get("ZSTD_LEVEL", "3"))
    strings = build_strings_from_adapt("allenai/tulu-3-sft-mixture")
    val_toks = build_tokens(strings)
    cx_val = _zstd_size(val_toks, level)
    console = Console()
    table = Table(title="mutual information via consistent Zstd streaming (symmetric)")
    table.add_column("Subset", style="cyan", no_wrap=True)
    table.add_column("Train Tokens", style="magenta", justify="right")
    table.add_column("Val Tokens", style="magenta", justify="right")
    table.add_column("Mutual Information", style="green", justify="right")
    subsets = get_dirs(base_path)
    tasks = [(base_path, s, val_toks, cx_val, 50_000_000, level) for s in subsets]
    with Pool(processes=min(len(tasks), max(1, cpu_count() // 2))) as pool:
        for subset, ntrain, mi in tqdm(pool.imap_unordered(process_subset, tasks), total=len(tasks), desc="Processing subsets"):
            table.add_row(subset, f"{ntrain:,}", f"{len(val_toks):,}", f"{mi:,} bytes")
    console.print(table)

if __name__ == "__main__":
    main()