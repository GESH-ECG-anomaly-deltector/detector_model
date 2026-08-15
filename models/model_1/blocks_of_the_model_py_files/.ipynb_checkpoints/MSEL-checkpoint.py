import math
import torch
import torch.nn as nn




class upper_MSEL(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        num_leads=12
        patch_len1=20
        patch_len2=40
        stride=patch_len1
        embed_dim=128
        max_len=1000
        #use_cls_token=True
        
        self.num_leads=num_leads
        self.patch_len1=patch_len1
        self.patch_len2=patch_len2
        self.stride=stride
        self.embed_dim=embed_dim
        self.max_len=max_len

        self.projection_1=nn.Conv1d(
            in_channels=self.num_leads,
            out_channels=self.embed_dim,
            kernel_size=self.patch_len1,
            stride=self.stride,
        )
        self.projection_2=nn.Conv1d(
            in_channels=self.num_leads,
            out_channels=self.embed_dim,
            kernel_size=self.patch_len2,
            stride=self.stride,
        )

        self.norm=nn.LayerNorm(self.embed_dim*2)
    
        self.cls_token=nn.Parameter(torch.zeros(1, 1, self.embed_dim*2))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        max_num_patches = (self.max_len-self.patch_len2)//self.stride  + 1
        pos_len = max_num_patches + 1     # (1 if use_cls_token else 0)
        self.pos_embed=nn.Parameter(torch.zeros(1, pos_len, self.embed_dim*2))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_leads, T)
        returns: (batch, num_patches [+1 if CLS], embed_dim)
        """

        batch, lead, T = x.shape

        z1=self.projection_1(x[:, : , 10:-10])
        z2=self.projection_2(x)
        z=torch.cat([z1, z2], dim=1)
        z=z.transpose(1, 2)
        z=self.norm(z)

        cls=self.cls_token.expand(batch, -1, -1)
        z=torch.cat([cls, z], dim=1)

        n_tokens=z.shape[1]
        z=z + self.pos_embed[:, :n_tokens, :]
        
        return z




class lower_MSEL(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        num_leads=12
        patch_len1=5
        patch_len2=100
        stride=patch_len1
        embed_dim=128
        max_len=1000
        #use_cls_token=True
        
        self.num_leads=num_leads
        self.patch_len1=patch_len1
        self.patch_len2=patch_len2
        self.stride=stride
        self.embed_dim=embed_dim
        self.max_len=max_len

        self.projection_1=nn.Conv1d(
            in_channels=self.num_leads,
            out_channels=self.embed_dim,
            kernel_size=self.patch_len1,
            stride=self.stride,
        )
        self.projection_2=nn.Conv1d(
            in_channels=self.num_leads,
            out_channels=self.embed_dim,
            kernel_size=self.patch_len2,
            stride=self.stride,
        )

        self.norm=nn.LayerNorm(self.embed_dim*2)
    
        self.cls_token=nn.Parameter(torch.zeros(1, 1, self.embed_dim*2))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        max_num_patches = (self.max_len-self.patch_len2)//self.stride  + 1
        pos_len = max_num_patches + 1     # (1 if use_cls_token else 0)
        self.pos_embed=nn.Parameter(torch.zeros(1, pos_len, self.embed_dim*2))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_leads, T)
        returns: (batch, num_patches [+1 if CLS], embed_dim)
        """

        batch, lead, T = x.shape

        z1=self.projection_1(x[:, : , 45:-50])
        z2=self.projection_2(x)
        z=torch.cat([z1, z2], dim=1)
        z=z.transpose(1, 2)
        z=self.norm(z)

        cls=self.cls_token.expand(batch, -1, -1)
        z=torch.cat([cls, z], dim=1)

        n_tokens=z.shape[1]
        z=z + self.pos_embed[:, :n_tokens, :]
        
        return z




