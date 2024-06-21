import torch
import os
import TinyStories.Trainer as ut
import numpy as np
from typing import List
from Classes.tokenizer import Tokenizer as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenizer(doc):
    tokenizer = T()
    tokens: List[int] = tokenizer.encode(s=doc, bos=True, eos=False)
    tokens_ts = torch.tensor(tokens, dtype=torch.long)  #
    assert (0 <= tokens_ts).all() and (tokens_ts < 2**14).all(), "token dictionary too large for uint16"
    return tokens_ts

gen_starting_text = "Once upon a time"
gen_toks = tokenizer(gen_starting_text).unsqueeze(0).to(device)

block_size = 256 
batch_size = 32
run_count = 0
batch_size_values = [40]
n_heads = 6
n_layers = 6
d_model = 768
dropout = 0.1
eval_iters = 10
max_lr = 6e-3
min_lr = max_lr * 0.1
warmup_steps = 150
max_steps = 57876 # 57876 steps is 1 epoch, if data is 10B tokens and batch size 0.5M tokens
vocab_size = 15_000 # next power of two doesn't increase performance

def calculate_total_batches(data, block_size, batch_size):
    return len(data) // (block_size * batch_size)

# load all file names from data
data_dir = 'data/tokenized_inputs'
files = os.listdir(data_dir)

total_batches = 0
for file in files:
    if file.endswith('npy'):
        pt_file = np.load(f'{data_dir}/{file}')
        n_batches = calculate_total_batches(data=pt_file, 
                                            block_size=block_size, 
                                            batch_size=batch_size)
        total_batches += n_batches

print(f'Total batches: {total_batches:,}')
steps = total_batches

trainer = ut.Trainer(vocab_size=vocab_size, block_size=block_size, dropout=dropout, 
                     n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                     device=device, learning_rate=max_lr, 
                     batch_size=batch_size, steps=max_steps, eval_iters=eval_iters)
for file in files:
    if file.endswith('npy'):
        trainer.load_data(f'{data_dir}/{file}', f'{data_dir}/tinystories_valid_000000.npy')
        x, y = trainer.make_batches(split='train')
        trainer.train_model()