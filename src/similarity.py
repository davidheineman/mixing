import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import gzip
import io, os
from rich.console import Console
from rich.table import Table

# def compress_array_as_bytes(arr: np.ndarray) -> int:
#     buffer = io.BytesIO()
#     with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
#         f.write(arr.tobytes())
#     return len(buffer.getvalue())

import blosc
def compress_array_as_bytes(arr: np.ndarray) -> int:
    blosc.set_nthreads(64)
    compressed = blosc.compress(
        arr.tobytes(), 
        typesize=arr.itemsize, 
        clevel=1, 
        shuffle=blosc.SHUFFLE,
        # cname='zstd'
        cname='lz4' # fast
    )
    return len(compressed)


def get_dirs(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs

def build_strings_from_eval(path):
    strings = []
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            strings.append(doc['request']['context'])
            strings.append(doc['request']['continuation'])


def build_strings_from_adapt(hf_path):
    """ Load an HF dataset as a list of strings """
    from datasets import load_dataset
    
    dataset = load_dataset(hf_path)
    strings = []
    
    for split in dataset:
        i = 0
        for example in tqdm(dataset[split], desc=f"Building {hf_path}"):
            if i > 10_000:
                return strings
            
            messages = example['messages']
            content_parts = []
            for message in messages:
                if 'content' in message:
                    content_parts.append(message['content'])
            if content_parts:
                strings.append(''.join(content_parts))

            i += 1
    
    return strings


def build_tokens(strings):
    text = '\n'.join(strings)

    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    tokens = tokenizer.encode(text)
    tokens = np.array(tokens, dtype=np.int32)

    return tokens


def main():
    base_path = "/oe-training-default/ai2-llm/preprocessed/dclm/baseline_topic_classified_sample"

    print("Pulling data...")

    # strings = build_strings_from_eval('data/arc_c.jsonl') # ARC-C approx. 3.5 MB
    
    # https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
    strings = build_strings_from_adapt('allenai/tulu-3-sft-mixture')

    print('Tokenizing...')

    val_toks = build_tokens(strings)

    console = Console()
    table = Table(title="mutual information using gzip compression")
    
    table.add_column("Subset", style="cyan", no_wrap=True)
    table.add_column("Train Tokens", style="magenta", justify="right")
    table.add_column("Val Tokens", style="magenta", justify="right")
    table.add_column("Mutual Information", style="green", justify="right")

    subsets = get_dirs(base_path)
    for subset in tqdm(subsets, desc="Processing subsets"):
        tqdm.write(f"Processing {subset}...")

        data_path = f'{base_path}/{subset}/dolma2-tokenizer/part-000-00000.npy'
        dtype = np.int32
        train_toks = np.memmap(data_path, mode="r", dtype=dtype)

        # truncate train toks
        # train_toks = train_toks[:500_000_000]
        train_toks = train_toks[:50_000_000]
        # train_toks = train_toks[:10_000_000]

        val_toks_size = compress_array_as_bytes(val_toks)
        train_toks_size = compress_array_as_bytes(train_toks)
        concat_size = compress_array_as_bytes(np.concatenate([val_toks, train_toks]))
        rev_concat_size = compress_array_as_bytes(np.concatenate([train_toks, val_toks]))

        mutual_info = val_toks_size + train_toks_size - concat_size

        # mutual_info = 0.5 * (
        #     (val_toks_size + train_toks_size - concat_size) +
        #     (val_toks_size + train_toks_size - rev_concat_size)
        # )

        # # Normalized Compression Distance (NCD)
        # # NCD(X,Y) = (C(XY) - min(C(X),C(Y))) / max(C(X),C(Y))
        # min_size = min(val_toks_size, train_toks_size)
        # max_size = max(val_toks_size, train_toks_size)
        # mutual_info = (concat_size - min_size) / max_size

        # \text{Info}(X \rightarrow Y) \approx C(Y) - \left[C(X \Vert Y) - C(X)\right]
        # mutual_info = train_toks_size - (concat_size - val_toks_size)

        table.add_row(
            subset,
            f"{len(train_toks):,}",
            f"{len(val_toks):,}",
            f"{mutual_info:,} bytes"
        )

    console.print(table)

if __name__ == '__main__':
    main()