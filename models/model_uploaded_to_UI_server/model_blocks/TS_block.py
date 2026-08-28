import torch.nn as nn
import torch
import math
from model_blocks.Transformer_Encoder import *






class upper_TS_block(nn.Module):
    def __init__(self, samples:int=256 , attention_blocks_count:int=8,
                 num_pre_layers:int=8, # num_post_layers:int=8, 
                 ema_start_layers:int=4, eta:float=0.5, lam:float=0.99):
        super().__init__()
        self.eta=eta
        self.lam=lam
        self.ema_start_layers=ema_start_layers

        #teb = transformer encode block
        layers_list=[]
        for i in range(num_pre_layers):
            layers_list.append( teb(samples, attention_blocks_count) )
        self.layers=nn.ModuleList(layers_list)

        #self.num_post_layers=num_post_layers
        

    def forward(self, x:torch.Tensor):
        ema_attention=None
        for i, layer in enumerate(self.layers):
            x, cls_attention=layer(x)
            if(i+1 >= self.ema_start_layers):
                if ema_attention is None:
                    ema_attention=cls_attention
                else:
                    ema_attention =  self.lam*ema_attention  +  (1-self.lam)*cls_attention

        patch_attention=ema_attention[:, 1:]
        m=patch_attention.shape[1]
        k = max( 1, int(m*self.eta) )

        top_k_idx = torch.topk(patch_attention, k, dim=-1).indices
        top_k_idx_sorted, _  = torch.sort(top_k_idx, dim=-1)

        batch_idx=torch.arange(x.shape[0], device=x.device).unsqueeze(-1)  #what?
        selected_batches=x[:, 1:, :][batch_idx, top_k_idx_sorted]
        cls_token=x[:, 0:1, :]
        x_selected=torch.cat( [cls_token, selected_batches], dim=1 )

        for layer in self.layers: #[:self.num_post_layers] : 
            x_selected, _ = layer(x_selected)

        return cls_token, x_selected[:, 0, :]  # first and final cls-token





class lower_TS_block(nn.Module):
    def __init__(self, samples:int=256 , attention_blocks_count:int=8,
                 num_pre_layers:int=5, #num_post_layers:int=5, 
                 ema_start_layers:int=3, eta:float=0.5, lam:float=0.99):
        super().__init__()
        self.eta=eta
        self.lam=lam
        self.ema_start_layers=ema_start_layers

        #teb = transformer encode block
        layers_list=[]
        for i in range(num_pre_layers):
            layers_list.append( teb(samples, attention_blocks_count) )
        self.layers=nn.ModuleList(layers_list)

        #self.num_post_layers=num_post_layers
        

    def forward(self, x:torch.Tensor):
        ema_attention=None
        for i, layer in enumerate(self.layers):
            x, cls_attention=layer(x)
            if(i+1 >= self.ema_start_layers):
                if ema_attention is None:
                    ema_attention=cls_attention
                else:
                    ema_attention =  self.lam*ema_attention  +  (1-self.lam)*cls_attention

        patch_attention=ema_attention[:, 1:]
        m=patch_attention.shape[1]
        k = max( 1, int(m*self.eta) )

        top_k_idx = torch.topk(patch_attention, k, dim=-1).indices
        top_k_idx_sorted, _  = torch.sort(top_k_idx, dim=-1)

        batch_idx=torch.arange(x.shape[0], device=x.device).unsqueeze(-1)  #what?
        selected_batches=x[:, 1:, :][batch_idx, top_k_idx_sorted]
        cls_token=x[:, 0:1, :]
        x_selected=torch.cat( [cls_token, selected_batches], dim=1 )

        for layer in self.layers: #[:self.num_post_layers] : 
            x_selected, _ = layer(x_selected)

        return cls_token , x_selected[:, 0, :]  #first and final cls-token