import torch, os
import numpy as np
import multiprocessing as mp
from typing import List 
from Classes.tokenizer import Tokenizer as T
from tqdm import tqdm # pip install tqdm

local_dir = "data/tokenized_inputs"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
shard_size = int(1e8)
tfw = open("data/train.csv", "r")
vfw = open("data/valid.csv", "r")

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokenizer = T()
    tokens: List[int] = tokenizer.encode(s=doc, bos=True, eos=False)
    tokens_np = np.array(tokens, dtype=np.int16)  # Use numpy.int16 instead of torch.long
    assert (0 <= tokens_np).all() and (tokens_np < 2**14).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def process_file(file, split):
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() - 1)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, file, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                filename = os.path.join(DATA_CACHE_DIR, f"tinystories_{split}_{shard_index:06d}.npy")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder
        # write any remaining tokens as the last shard
        if token_count != 0:
            filename = os.path.join(DATA_CACHE_DIR, f"tinystories_{split}_{shard_index:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])

# Process both training and validation files
process_file(vfw, "valid")
process_file(tfw, "train")