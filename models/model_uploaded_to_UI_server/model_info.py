## load libraries
import numpy as np
import os
import shutil
import math

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import (
    MultilabelF1Score, 
    MultilabelRecall,
    MultilabelAUROC,
    MultilabelAccuracy)

from model_blocks.CNN_blocks import *
from model_blocks.MSEL import *
from model_blocks.Transformer_Encoder import *
from model_blocks.TS_block import *

from torchinfo import summary
print("packages and libraries are imported.\n")




device = "cpu"
print("device:", device)



## Model Definition
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN_blocks1=CNN_blocks1()
        self.CNN_blocks2=CNN_blocks2()

        self.lower_MSEL=lower_MSEL()
        self.lower_TS_block=lower_TS_block()  ###### parameters are remained
        
        self.upper_MSEL=upper_MSEL()
        self.upper_TS_block=upper_TS_block() ####### parameters are remained
        
        self.network=nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(), 
            nn.Dropout(0.3),
            
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(80, 8)
        )

        

    def forward(self, x):
        
        cnn_output1=self.CNN_blocks1(x)
        cnn_output2=self.CNN_blocks2(cnn_output1)
        
        upper_MSEL_out         =  self.upper_MSEL(cnn_output2)
        upper_cls1, upper_cls2 =  self.upper_TS_block(upper_MSEL_out)
        
        lower_MSEL_out         =  self.lower_MSEL(cnn_output2)
        lower_cls1, lower_cls2 =  self.lower_TS_block(lower_MSEL_out)
        
        net_in    =  torch.cat([lower_cls2, upper_cls2], dim=1)
       # classification_output  =  torch.sigmoid( self.network(net_in) )
        
       # return classification_output, [upper_cls1, lower_cls1, upper_cls2, lower_cls2]
        return self.network(net_in)  , [upper_cls1, lower_cls1, upper_cls2, lower_cls2]
        

print("classifier() is defined.\n")




model=Classifier().to(device)

print("model is gone to device.\n")




checkpoint = torch.load(
    "checkpoint_manager/best_checkpoint.pt",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("\n\n")
summary(model, input_shape=(1, 12, 1000))
