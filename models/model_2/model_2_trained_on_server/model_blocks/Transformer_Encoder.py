import torch.nn as nn
import torch
import math






#mhsa = multi-head self attention
class mhsa(nn.Module):
    def __init__(self, d_model: int, num_heads: int):  #d_model:input_length             num_heads: attention_blocks_count
        super().__init__()

        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim= d_model//num_heads

        self.q_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        #seq_len = number of tokens
        
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

        cls_attn=attn_weights[:, :, 0, :]  
        cls_attn=cls_attn.mean(dim=1)
        out=self.out_proj(attn_output)

        return out, cls_attn




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
    def __init__(self, samples: int, attention_blocks_count: int):
        #samples=256  attention_blocks_count=8
        d_model=samples        
        num_heads=attention_blocks_count
        d_ff=d_model*4
        
        super().__init__()

        self.norm1=nn.LayerNorm(d_model)
        self.self_attention=mhsa(d_model, num_heads)
        self.norm2=nn.LayerNorm(d_model)
        self.mlp=mlp(d_model, d_ff)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n1=self.norm1(x)
        att_out, cls_attn=self.self_attention(x_n1)
        
        tmp=x + att_out
        
        x_n2=self.norm2(tmp)
        mlp_out=self.mlp(x_n2)

        teb_out=tmp + mlp_out

        return teb_out, cls_attn






