import torch
import torch.nn as nn
import math




#mhsa = multi-head self attention
class mhsa(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self.d_modle=d_model
        self.num_heads=num_heads
        self.head_dim= d_model//num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        k=k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q=q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v=v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores=torch.matmul(q, k.transpose(-2, -1)) /math.sqrt(self.head_dim)

        attn_weights=torch.softmax(scores, dim=-1)
        attn_output=torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(attn_output)




#multi layer perceptron
class mlp(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(), 
            nn.Linear(d_ff, d_model)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_out=self.net(x)
        return mlp_out




#Transformer Encoder Block
class teb(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):  # d_ff=2048
        super().__init__()

        self.norm1=nn.LayerNorm(d_model)
        self.self_attention=mhsa(d_model, num_heads)
        self.norm2=nn.LayerNorm(d_model)
        self.mlp=mlp(d_model, d_ff)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n1=self.self.norm1(x)
        att_out=self.self_attention(x_n1)
        
        tmp=x + att_out
        
        x_n2=self.norm2(tmp)
        mlp_out=self.mlp(x_n2)

        teb_out=tmp + mlp_out

        return teb_out

        