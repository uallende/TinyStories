import torch
import os
import TinyStories.Trainer as ut
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_total_batches(data, block_size, batch_size):
    return len(data) // (block_size * batch_size)

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


block_size = 256+32
batch_size = 32
run_count = 0
batch_size_values = [40]
n_heads = 8
n_layers = 8
d_model = 768
dropout = 0.1

epochs = 8192
eval_iters = 10

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 56400 # 56400 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
vocab_size = 15_000 # next power of two doesn't increase performance

trainer = ut.Trainer(vocab_size=vocab_size, block_size=block_size, dropout=dropout, 
                     n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                     device=device, learning_rate=max_lr, 
                     batch_size=batch_size, steps=150, eval_iters=eval_iters)

# load all file names from data
data_dir = 'data/tokenized_inputs'
files = os.listdir(data_dir)

for file in files:
    if file.startswith('tns'):
        trainer.load_data(f'{data_dir}/{file}', 'data/tokenized_inputs/val.pt')
        x, y = trainer.make_batches(split='train')
        trainer.train_model()
