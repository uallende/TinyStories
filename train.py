import torch
import os
import Classes.myGPT
import TinyStories.utils_hp_search as ut

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_total_batches(data, block_size, batch_size):
    return len(data) // (block_size * batch_size)

trainer = ut.Trainer(vocab_size=100, block_size=10, dropout=0.1, dff=100, n_layers=1, d_model=100, n_heads=1,
                     device=device, learning_rate=0.001, batch_size=32, epochs=10, eval_iters=100)

trainer.load_data('data/tokenized_inputs/tns_chunk_0.pt', 'data/tokenized_inputs/val.pt')
x, y = trainer.make_batches('train')

# load all file names from data
data_dir = 'data/tokenized_inputs'
files = os.listdir(data_dir)

steps = 1
for file in files:
    if file.startswith('tns'):
        train = torch.load(data_dir + file)
        total_batches = calculate_total_batches(train, block_size, batch_size)
        train_dl = make_batches(train, block_size, batch_size)

        for epoch, (Xb, Yb) in enumerate(tqdm(train_dl, total=total_batches)):

            for param_group in optimizer.param_groups:
                param_group['lr'] = get_lr(steps)

            Xb, Yb = Xb.to(device), Yb.to(device)
            logits, loss = m(Xb, Yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss, steps)
            steps += 1

            if (epoch+1) % 100 == 0:
                _, val_loss = estimate_loss(m, train, val, block_size, batch_size, eval_iters)
                writer.add_scalar('Loss/val', val_loss, steps)

        train_loss, val_loss = estimate_loss(m, train, val, block_size, batch_size, eval_iters)

        if steps >= max_iters:
            break

# save torch model
torch.save(m.state_dict())