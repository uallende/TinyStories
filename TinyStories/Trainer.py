import torch, math, time, os, sys, inspect
import numpy as np
from Classes.myGPT import Model  
from torch.utils.tensorboard import SummaryWriter
from typing import List

sys.path.append('../')  

from Classes.tokenizer import Tokenizer as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, vocab_size, block_size, 
                 dropout, n_layers, d_model, n_heads,
                device, learning_rate, batch_size,
                steps, eval_iters):
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout = dropout
        self.dff = d_model * 3
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        self.batch_size = batch_size
        self.steps = steps
        self.eval_iters = eval_iters
        self.max_lr = learning_rate
        self.min_lr = self.max_lr * 0.1
        self.warmup_steps = 1000
        self.max_steps = self.steps 
        self.tkrz = T()
        self.gen_starting_text = "Once upon a time"
        self.gen_toks = self.tokenizer(self.gen_starting_text).unsqueeze(0).to(device)

        self.m = Model(vocab_size=self.vocab_size, block_size=self.block_size,
                       dropout=self.dropout, dff=self.dff, n_layers=self.n_layers,
                       d_model=self.d_model, n_heads=self.n_heads).to(self.device)
        
        # self.m = torch.compile(self.m) not available 1060

    def tokenizer(self, doc):
        tokens: List[int] = self.tkrz.encode(s=doc, bos=True, eos=False)
        tokens_ts = torch.tensor(tokens, dtype=torch.long)  #
        assert (0 <= tokens_ts).all() and (tokens_ts < 2**14).all(), "token dictionary too large for uint16"
        return tokens_ts
                
    def load_data(self, train, val):
        self.train = self._convert_to_tensor(train)
        self.val = self._convert_to_tensor(val)

    def save_config(self):
        # Filter __dict__ if necessary to remove non-config attributes
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k != 'm'}
        return config_dict

    def _convert_to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):  # Assume filepath
            return torch.tensor(np.load(data), dtype=torch.long)
        elif isinstance(data, (list, np.ndarray)):
            return torch.tensor(data)
        elif isinstance(data, (np.ndarray)):
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):

        param_dict = {pn: p for pn, p in self.m.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    @torch.no_grad()
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
    
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_steps:
            return self.max_lr * (it+1) / self.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.max_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    
    def train_model(self):
        writer = SummaryWriter(f'runs/heads_{self.n_heads}_layers_{self.n_layers}_dmodel_{self.d_model}_batch_size_{self.batch_size}')
        log_dir = "model_checkpoints"
        
        n_params = sum(p.nelement() for p in self.m.parameters())
        print(f'Number of parameters: {n_params:,}')
        print(f'Tokens per batch: {self.block_size*self.batch_size}')
        optimizer = self.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=self.device)
        
        grad_accum_steps = 8
        global_step = 0

        for step in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            st = time.time()
            last_step = (step == self.steps - 1)

            for accum_step in range(grad_accum_steps):

                Xb, Yb = self.make_batches(split='train')            
                _, loss = self.m(Xb, Yb)  # forward pass
                loss = loss / grad_accum_steps  # scale loss
                loss_accum += loss.detach().item()  # Accumulate the scaled loss
                loss.backward()  # backward pass and accumulate gradients
                
                # Log less frequently
                if global_step % 10 == 0:
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                global_step += 1  # increment global step for each sub-step

            norm = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)  # Clip gradients
            lr = self.get_lr(global_step)  # Get current learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr  # Update learning rate
            optimizer.step()  # Apply accumulated gradients

            et = time.time()
            dt = (et - st)
            n_tokens = self.batch_size * self.block_size * grad_accum_steps  # Adjust for accumulated steps
            print(f"step: {step+1}. time {dt*1000:.3f} ms. "
                  f"{n_tokens/dt:,.0f} tok/sec | train loss: {loss_accum:.3f}. "
                  f"lr:{lr:.3e}. "
                  f"norm: {norm:.2f}")

            # Evaluation and checkpoint saving
            if step % 100 == 99:
                l = self.estimate_loss()
                writer.add_scalar('Loss/val', l['val'], step)
                print(f"Step: {step+1}. val loss: {l['val']:.3f}. train loss: {l['train']:.3f}")

                generated_idx = self.m.generate(self.gen_toks, max_new_tokens=30)
                generated_text = self.tkrz.decode(generated_idx[0].tolist())
                print(f"Generated text: {self.gen_starting_text} {generated_text}")

            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': self.m.state_dict(),
                    'config': self.save_config(),
                    'step': step,
                    'opt': optimizer.state_dict(),
                    'val_loss': l['val']
                }
                torch.save(checkpoint, checkpoint_path)

# # # logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
# # # logging.debug(f"block_size type: {type(block_size)}, block_size value: {block_size}")
# # # logging.info(f"Data type: {type(data)}, Data shape: {data.shape}")
