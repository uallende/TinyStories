import torch
import os
import TinyStories.utils_hp_search as ut

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_total_batches(data, block_size, batch_size):
    return len(data) // (block_size * batch_size)

block_size = 512
batch_size = 16
run_count = 0
batch_size_values = [40]
n_heads = 8
n_layers = 8
d_model = 768
dropout = 0.1
learning_rate = 3e-4
epochs = 8192
eval_iters = 10

vocab_size = 15_000 # next power of two doesn't increase performance

trainer = ut.Trainer(vocab_size=vocab_size, block_size=block_size, dropout=dropout, 
                     n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                     device=device, learning_rate=learning_rate, 
                     batch_size=batch_size, epochs=50, eval_iters=2)

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