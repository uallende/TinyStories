import torch
import sys
import numpy as np
sys.path.append('../')  
from Classes.myGPT import Model  
from torch.utils.tensorboard import SummaryWriter

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
        
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.dff = self.d_model * 4  # Assuming dff is always 4 * d_model
        self.reinitialize_model()
    
    def reinitialize_model(self):
        self.m = Model(vocab_size=self.vocab_size, block_size=self.block_size,
                       dropout=self.dropout, dff=self.dff, n_layers=self.n_layers,
                       d_model=self.d_model, n_heads=self.n_heads).to(self.device)

    def load_data(self, train, val):
        self.train = self._convert_to_tensor(train)
        self.val = self._convert_to_tensor(val)

    def _convert_to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):  # Assume filepath
            return torch.load(data)
        elif isinstance(data, (list, np.ndarray)):
            return torch.tensor(data)
        else:
            raise ValueError("Unsupported data type")

    def make_batches(self, split=None):
        
        data = self.train if split == 'train' else self.val
        ix = torch.randint(len(data) - self.block_size, (self.batch_size, ))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+self.block_size] for i in ix])
        x, y = x.to(self.device), y.to(self.device)

        return x, y

    def estimate_loss(self):
        out = {}
        self.m.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.make_batches()
                logits, loss = self.m(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.m.train()

        return out

    def train_model(self):
        writer = SummaryWriter(f'runs/heads_{self.n_heads}_layers_{self.n_layers}_dmodel_{self.d_model}_batch_size_{self.batch_size}')
        # writer = SummaryWriter(f'runs/dropout_{self.dropout}_block_size_{self.block_size}_learning_rate_{self.learning_rate}')                
        optimizer = torch.optim.AdamW(self.m.parameters(), lr=self.learning_rate)
        n_params = sum(p.nelement() for p in self.m.parameters())
        print(f'Number of parameters: {n_params:,}')

        for epoch in range(self.epochs):

            Xb, Yb = self.make_batches(split='train')            
            logits, loss = self.m(Xb, Yb) # B, C
            writer.add_scalar('Loss/train', loss, epoch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 99:
                l = self.estimate_loss()
                writer.add_scalar('Loss/val', l['val'], epoch)
                #writer.add_scalar('Loss/train', l['train'], epoch)

            if epoch % 250 == 249:
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