from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import tiktoken


root = Path(__file__)
files = Path(root.parent.parent / "data/preprocessed").glob("*.txt")
num_proc = 4
enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out


def build_file(tokenized):
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'])
        outfile = root.parent.parent / f"data/processed/{split}.bin"
        filename = outfile.as_posix()
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        print(f"writing {filename}...")
        idx = 0
        for example in tqdm(dset):
            arr[idx: idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()


def main():
    dataset = load_dataset("text", data_files=[file.as_posix() for file in files])
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    build_file(tokenized)


if __name__ == '__main__':
    main()