import torch
import os
import TinyStories.Trainer as ut
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_total_batches(data, block_size, batch_size):
    return len(data) // (block_size * batch_size)

block_size = 512
batch_size = 8
run_count = 0
batch_size_values = [40]
n_heads = 4
n_layers = 4
d_model = 768
dropout = 0.1

epochs = 8192
eval_iters = 10

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 56400 # 56400 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

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

vocab_size = 15_000 # next power of two doesn't increase performance

trainer = ut.Trainer(vocab_size=vocab_size, block_size=block_size, dropout=dropout, 
                     n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                     device=device, learning_rate=max_lr, 
                     batch_size=batch_size, steps=150, eval_iters=2)

trainer.load_data('data/tokenized_inputs/tns_chunk_0.pt', 'data/tokenized_inputs/val.pt')
x, y = trainer.make_batches('train')

# load all file names from data
data_dir = 'data/tokenized_inputs'
files = os.listdir(data_dir)

steps = 1
for file in files:
    if file.startswith('tns'):
        trainer.load_data(f'{data_dir}/{file}', 'data/tokenized_inputs/val.pt')
        total_batches = calculate_total_batches(trainer.train, block_size, batch_size)
        x, y = trainer.make_batches(split='train')

        # print(x[:15], y[:15])
        trainer.train_model()
        # for epoch, (Xb, Yb) in enumerate(tqdm(train_dl, total=total_batches)):

        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = get_lr(steps)

        #     Xb, Yb = Xb.to(device), Yb.to(device)
        #     logits, loss = m(Xb, Yb)
        #     optimizer.zero_grad(set_to_none=True)
        #     loss.backward()
        #     optimizer.step()
        #     writer.add_scalar('Loss/train', loss, steps)
        #     steps += 1

        #     if (epoch+1) % 100 == 0:
        #         _, val_loss = estimate_loss(m, train, val, block_size, batch_size, eval_iters)
        #         writer.add_scalar('Loss/val', val_loss, steps)

        # train_loss, val_loss = estimate_loss(m, train, val, block_size, batch_size, eval_iters)

        # if steps >= max_iters:
        break

import sys; sys.exit()

# save torch model
torch.save(m.state_dict())