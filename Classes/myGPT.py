import torch
from torch.nn import functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, d_model, block_size, dropout=0.1):

        super().__init__()
        assert d_model % n_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.n_heads = n_heads
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.att_proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size).bool(), diagonal=1))

    def forward(self, x):

        q = x
        k = x
        v = x
        #print("Shape of x inside model:", x.shape)
        B,T,_ = x.shape 

        dk = self.d_model // self.n_heads

        # linear projections
        q = self.query(q) 
        k = self.key(k) 
        v = self.value(v) 

        # add number of heads
        q = q.view(B,T,self.n_heads,dk).permute(0,2,1,3)   # B,T,h,dk
        k = k.view(B,T,self.n_heads,dk).permute(0,2,1,3)  
        v = v.view(B,T,self.n_heads,dk).permute(0,2,1,3)  
        
        # attention 
        x = q @ k.transpose(-2,-1) # B,h,T,dk @ B,h,dk,T --> B,h,T,T
        x = x * dk ** -0.5 # B,h,T,T
        x = x.masked_fill(self.mask, float('-inf')) # B,h,T,T
        x = F.softmax(x, dim=(-1)) # B,n_h,T,T 
        x = x @ v  # B,h,T,T @ B,T,h,dv --> B,h,T,dv
        B,h,T,dv = x.shape
        x = x.transpose(2,1).contiguous().view(B,T,h*dv) #B,T,C
        out = self.att_proj(x) # B,T,C

        return out
    
class AttentionLayer(nn.Module):
    def __init__(self, n_heads, d_model, block_size, dropout):
        super().__init__()

        self.att = MultiHeadAttention(n_heads, d_model, block_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.att(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.seq = nn.Sequential(
                    nn.Linear(d_model, dff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dff, d_model)
                    )

    def forward(self, x):
        x = self.seq(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, block_size, dropout, dff) :
        super().__init__()

        self.att = AttentionLayer(n_heads, d_model,
                                  block_size, dropout)
        
        self.ffw = FeedForward(d_model, dff, dropout)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x + self.att(self.lnorm1(x))
        x = x + self.ffw(self.lnorm2(x))

        return x
    
class Model(nn.Module):

    def __init__(self, vocab_size, block_size, dropout, 
                 dff, n_layers, d_model, n_heads) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.embedding_table = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)

        self.decoder = nn.Sequential(*[DecoderLayer(n_heads,
                                                    d_model,
                                                    block_size,
                                                    dropout,
                                                    dff) 
                                                    for _ in range(n_layers)])
        
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):

        #print(x.shape)
        embeds = self.embedding_table(x)
        positions = self.pos_embedding(torch.arange(self.block_size, device=device))
        x = embeds + positions
        x = self.decoder(x)
        logits = self.out(x)

        if targets == None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            #print(logits.shape, targets.shape)
            loss = F.cross_entropy(input=logits, target=targets)

        return logits, loss    
    
    def generate(self, idx, max_new_tokens, top_k = 30):
        # idx is (B, T) array of indices in the current context
        B, T = idx.shape
        if T < self.block_size:
          idx = F.pad(idx, (0, self.block_size - T))

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            top_k_values, top_k_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_k_values, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx_next = top_k_indices.gather(-1, idx_next)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx[:,T:]
