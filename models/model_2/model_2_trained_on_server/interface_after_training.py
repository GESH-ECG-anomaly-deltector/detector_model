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
print("packages and libraries are imported.\n")




device = "cpu"
print("device:", device)




## load data:
x_test=np.load("dataset/x_test.npy")
y_test=np.load("dataset/y_test.npy")


mean=np.load("dataset/train_mean.npy")
std= np.load("dataset/train_std.npy")

x_test  = (x_test  -mean) / (std + 1e-8)
x_test  = torch.tensor(x_test,  dtype=torch.float)

y_test  = torch.tensor(y_test,  dtype=torch.long)

test_dataset  = TensorDataset(x_test, y_test)  
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)  




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

outputs=[]
with torch.no_grad():
    for x, y in test_loader:
        outputs.append(model(x.to(device))[0])
#output=(torch.cat(outputs), None)
output= (torch.cat(outputs), None)


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

TP= ((y_test_pred>0.5) & (y_test==1)).sum(dim=0)
FP= ((y_test_pred>0.5) & (y_test==0)).sum(dim=0)
TN= ((y_test_pred<0.5) & (y_test==0)).sum(dim=0)
FN= ((y_test_pred<0.5) & (y_test==1)).sum(dim=0)
print("\n\n")
print("TP:\t\t",   TP[0], "\t", TP[1], "\t", TP[2], "\t", TP[3], 
                   TP[4], "\t", TP[5], "\t", TP[6], "\t", TP[7] )

print("FP:\t\t",   FP[0], "\t", FP[1], "\t", FP[2], "\t", FP[3],
                   FP[4], "\t", FP[5], "\t", FP[6], "\t", FP[7] )

print("TN:\t\t",   TN[0], "\t", TN[1], "\t", TN[2], "\t", TN[3],
                   TN[4], "\t", TN[5], "\t", TN[6], "\t", TN[7] )

print("FN:\t\t",   FN[0], "\t", FN[1], "\t", FN[2], "\t", FN[3],
                   FN[4], "\t", FN[5], "\t", FN[6], "\t", FN[7] )

