import torch
import os
import TinyStories.Trainer as ut
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

block_size = 256+32
batch_size = 32
run_count = 0
batch_size_values = [40]
n_heads = 8
n_layers = 8
d_model = 768
dropout = 0.1
eval_iters = 10
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 500
max_steps = 57876 # 57876 steps is 1 epoch, if data is 10B tokens and batch size 0.5M tokens
vocab_size = 15_000 # next power of two doesn't increase performance

def calculate_total_batches(data, block_size, batch_size):
    return len(data) // (block_size * batch_size)

# load all file names from data
data_dir = 'data/tokenized_inputs'
files = os.listdir(data_dir)

total_batches = 0
for file in files:
    if file.startswith('tns'):
        pt_file = torch.load(f'{data_dir}/{file}')
        n_batches = calculate_total_batches(data=pt_file, 
                                            block_size=block_size, 
                                            batch_size=batch_size)
        total_batches += n_batches

print(f'Total batches: {total_batches:,}')
steps = total_batches

trainer = ut.Trainer(vocab_size=vocab_size, block_size=block_size, dropout=dropout, 
                     n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                     device=device, learning_rate=max_lr, 
                     batch_size=batch_size, steps=steps, eval_iters=eval_iters)
for file in files:
    if file.startswith('tns'):
        trainer.load_data(f'{data_dir}/{file}', 'data/tokenized_inputs/val.pt')
        x, y = trainer.make_batches(split='train')
        trainer.train_model()
