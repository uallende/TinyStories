import torch
import sys
import numpy as np
sys.path.append('../')  
from Classes.myGPT import Model  
from torch.utils.tensorboard import SummaryWriter
import itertools

class Trainer:
    def __init__(self, vocab_size, block_size, 
                 dropout, dff, n_layers, d_model, n_heads,
                device, learning_rate, batch_size,
                epochs, eval_iters):
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout = dropout
        self.dff = dff
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_iters = eval_iters

        self.m = Model(vocab_size=self.vocab_size, block_size=self.block_size,
                       dropout=self.dropout, dff=self.dff, n_layers=self.n_layers,
                       d_model=self.d_model, n_heads=self.n_heads).to(self.device)
        
    @torch.no_grad()
    def estimate_loss(self, train_dl, val_dl):
        def compute_average_loss(data_loader, split):
            losses = []
            for i, (X, Y) in enumerate(itertools.islice(data_loader, self.eval_iters)):
                X, Y = X.to(self.device), Y.to(self.device)
                _, loss = self.m(X, Y)
                losses.append(loss.item())
            return sum(losses) / len(losses)

        self.m.eval()
        train_loss = compute_average_loss(train_dl, 'train')
        val_loss = compute_average_loss(val_dl, 'val')
        self.m.train()
        return {'train': train_loss, 'val': val_loss}

    def train_model(self, Xb, Yb, train_dl, val_dl):
        # writer = SummaryWriter(f'runs/heads_{self.n_heads}_layers_{self.n_layers}_dmodel_{self.d_model}_batch_size_{self.batch_size}')
        # writer = SummaryWriter(f'runs/dropout_{self.dropout}_block_size_{self.block_size}_learning_rate_{self.learning_rate}')                
        optimizer = torch.optim.AdamW(self.m.parameters(), lr=self.learning_rate)
        n_params = sum(p.nelement() for p in self.m.parameters())
        print(f'Number of parameters: {n_params:,}')

        for epoch in range(self.epochs):
            Xb, Yb = Xb.to(self.device), Yb.to(self.device)
            logits, loss = self.m(Xb, Yb) # B, C
            #writer.add_scalar('Loss/train', loss, epoch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if epoch % 25 == 24:
                l = self.estimate_loss(train_dl, val_dl)
                print(f'Epoch: {epoch+1}. Loss: {l["train"]:.3f}. Loss: {l["val"]:.3f}')
               

# logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
# train = torch.load('train.pt')
# val = torch.load('val.pt')

# def make_batches(block_size:int, 
#                  batch_size:int, train:torch.tensor,
#                  val:torch.tensor,
#                  device, split=None):

#     logging.debug(f"block_size type: {type(block_size)}, block_size value: {block_size}")

#     data = train if split == 'train' else val
#     logging.info(f"Data type: {type(data)}, Data shape: {data.shape}")

#     ix = torch.randint(len(data) - block_size, (batch_size, ))
#     logging.debug(f"Index type: {type(ix)}, Index shape: {ix.shape}")

#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+1+block_size] for i in ix])

#     logging.debug(f"x type: {type(x)}, x shape: {x.shape}")
#     logging.debug(f"y type: {type(y)}, y shape: {y.shape}")
#     x, y = x.to(device), y.to(device)
    
#     return x, y

# @torch.no_grad()
# def self.estimate_loss(m, self.eval_iters,
#                   block_size, 
#                   batch_size, 
#                   train, val, 
#                   device):
#     out = {}
#     m.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(self.eval_iters)
#         for k in range(self.eval_iters):
#             X, Y = make_batches(split=split,
#                                 block_size=block_size, 
#                                 batch_size=batch_size, 
#                                 train=train, 
#                                 val=val, 
#                                 device=device,
#                                 )
#             logits, loss = m(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     m.train()
#     return out

# def train_model(vocab_size, block_size, dropout,
#                 dff, n_layers, d_model, n_heads,
#                 device, learning_rate, batch_size,
#                 epochs, self.eval_iters):

#     writer = SummaryWriter(f'runs/heads_{n_heads}_layers_{n_layers}_dmodel_{d_model}')

#     m = Model(vocab_size=vocab_size, block_size=block_size, 
#                       dropout=dropout, dff=dff, n_layers=n_layers,
#                       d_model=d_model, n_heads=n_heads).to(device)
    
#     optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
#     n_params = sum(p.nelement() for p in m.parameters())
#     print(f'Number of parameters: {n_params:,}')

#     for epoch in range(epochs):

#         Xb, Yb = make_batches(split='train',
#                               batch_size=batch_size,
#                               block_size=block_size,
#                               train=train,
#                               val=val,
#                               device=device)
        
#         logits, loss = m(Xb, Yb) # B, C

#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()

#         if epoch % 10 == 9:
#             l = self.estimate_loss(m=m,
#                               self.eval_iters=self.eval_iters,
#                               block_size=block_size,
#                               batch_size=batch_size,
#                               train=train,
#                               val=val,
#                               device=device)
            
#             writer.add_scalar('Loss/val', l['val'], epoch)
#             writer.add_scalar('Loss/train', l['train'], epoch)

#         final_metrics = self.estimate_loss(  m=m,     
#                                         self.eval_iters=self.eval_iters,
#                                         block_size=block_size,
#                                         batch_size=batch_size,
#                                         train=train,
#                                         val=val,
#                                         device=device)
        
#     hparams = {'d_model': d_model, 'n_heads': n_heads, 'n_layers': n_layers}
#     writer.add_hparams(hparams, {'Loss/val': final_metrics['val']})
#         # print(f"Iteration {epoch}. Training Loss: {l['train']:.3f}. Evaluation Loss: {l['val']:.3f}")