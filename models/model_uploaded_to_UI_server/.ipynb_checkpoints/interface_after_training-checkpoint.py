## load libraries
import numpy as np
import os
import shutil
import math
print("numpy, os, shutil, and math are imported.\n")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
print("train_test_split, and accuracy_score are imported.\n")

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchmetrics.classification import (
    MultilabelF1Score, 
    MultilabelRecall,
    MultilabelAUROC,
    MultilabelAccuracy)
from torch.utils.tensorboard import SummaryWriter
print("torch, nn, optim, TensorDataset, DataLoader, radom_split, metrics, and SummaryWriter are imported.\n")

from CNN_blocks import *
from MSEL import *
from Transformer_Encoder import *
from TS_block import *
print("my codes are imported.\n")




device = "cpu"
print("device:", device)




## load data:
x_test=np.load("x_test.npy")
y_test=np.load("y_test.npy")


mean=np.load("train_mean.npy")
std= np.load("train_std.npy")

x_test  = (x_test  -mean) / (std + 1e-8)
x_test  = torch.tensor(x_test,  dtype=torch.float)

y_test  = torch.tensor(y_test,  dtype=torch.long)




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
            
            nn.Linear(100, 80),
            nn.ReLU(),
            
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
        
        return self.network( net_in ), [upper_cls1, lower_cls1, upper_cls2, lower_cls2]


print("classifier() is defined.\n")




model=Classifier().to(device)

print("model is gone to device.\n")




## metrics
num_labels=8
f1_micro=MultilabelF1Score(num_labels=num_labels, average='micro', threshold=0.5)
f1_macro=MultilabelF1Score(num_labels=num_labels, average='macro', threshold=0.5)
f1_per_label=MultilabelF1Score(num_labels=num_labels, average=None, threshold=0.5)

recall_micro = MultilabelRecall(num_labels=num_labels, average='micro', threshold=0.5)
recall_macro = MultilabelRecall(num_labels=num_labels, average='macro', threshold=0.5)
recall_per_label = MultilabelRecall(num_labels=num_labels, average=None, threshold=0.5)

auroc_per_label = MultilabelAUROC(num_labels=num_labels, average=None)

acc_per_label = MultilabelAccuracy(num_labels=num_labels, average=None, threshold=0.5)
acc_micro = MultilabelAccuracy(num_labels=num_labels, average='micro', threshold=0.5)
acc_macro = MultilabelAccuracy(num_labels=num_labels, average='macro', threshold=0.5)

f1_micro = f1_micro.to(device)
f1_macro = f1_macro.to(device)
f1_per_label = f1_per_label.to(device)
recall_micro = recall_micro.to(device)
recall_macro = recall_macro.to(device)
recall_per_label = recall_per_label.to(device)
auroc_per_label = auroc_per_label.to(device)
acc_per_label = acc_per_label.to(device)
acc_micro = acc_micro.to(device)
acc_macro = acc_macro.to(device)




print("\n\n*** *** *** *** *** ***")
print("test evaluation:")
checkpoint = torch.load(
    "checkpoint_manager/best_checkpoint.pt",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Best checkpoint loaded.")
print("Best checkpoint epoch:", checkpoint["epoch"])
print("Best validation loss:", checkpoint["custom_paper's_loss"], "\n")


with torch.no_grad():
    output = model(x_test)
y_test_pred=torch.sigmoid(output[0])
#print(y_test_pred)
print("f1_micro:        \t", f1_micro(y_test_pred, y_test))
print("f1_per_label:    \t", f1_per_label(y_test_pred, y_test))
print("f1_macro:        \t", f1_macro(y_test_pred, y_test), "\n")

print("recall_micro:    \t", recall_micro(y_test_pred, y_test))
print("recall_macro:    \t", recall_macro(y_test_pred, y_test))
print("recall_per_label:\t", recall_per_label(y_test_pred, y_test), "\n")

print("auroc_per_label: \t", auroc_per_label(y_test_pred, y_test), "\n")

print("acc_per_label:   \t", acc_per_label(y_test_pred, y_test))
print("acc_micro:       \t", acc_micro(y_test_pred, y_test))
print("acc_macro:       \t", acc_macro(y_test_pred, y_test), "\n")