import torch
import sys
import numpy as np
sys.path.append('../')  
from Classes.myGPT import Model  
from torch.utils.tensorboard import SummaryWriter
import logging
import itertools  

logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

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
                logging.info(f'{split}: {X[0][:5]}, {Y[0][:5]}')
            return sum(losses) / len(losses)

        self.m.eval()
        train_loss = compute_average_loss(train_dl, 'train')
        val_loss = compute_average_loss(val_dl, 'val')
        self.m.train()
        logging.info(f"Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
        return {'train': train_loss, 'val': val_loss}

    
    def load_model(self):
        self.optimizer = torch.optim.AdamW(self.m.parameters(), lr=self.learning_rate)
        n_params = sum(p.nelement() for p in self.m.parameters())
        logging.info(   f"\n\nNumber of parameters: {n_params:,} \n"
                        f"Number of layers: {self.n_layers} \n"
                        f"Number of heads: {self.n_heads} \n"
                        f"Block size: {self.block_size} \n"
                        f"Batch size: {self.batch_size} \n"
                        f"Learning rate: {self.learning_rate} \n"
                        f"Eval iters: {self.eval_iters} \n"
                        f"Dropout: {self.dropout} \n"
                        f"Vocab size: {self.vocab_size} \n"
                        f"Device: {self.device} \n")

    def forward_pass(self, Xb, Yb):
        logits, loss = self.m(Xb, Yb)
        return logits, loss
    
    def backward_pass(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()