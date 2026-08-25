## load libraries
import numpy as np
import os
import shutil
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from load_dataset import *

from CNN_blocks import *
from MSEL import *
from Transformer_Encoder import *
from TS_block import *




device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
#device = "cpu"
print(device)




## make dataset

#must be executed first in previous directories:    
#wget https://physionet.org/content/ecg-arrhythmia/get-zip/1.0.0/

if not "dataset" in os.listdir():
    ldob=load_dataset()
    ldob.load_dataset()
    ldob.denoise_signals()
    ldob.reduce_sample_rate()
    ldob.define_labels()
    ldob.define_8_superclasses()
    ldob.one_hot_encodding()

    signals, labels=ldob.get_signals_and_labels()

    if "dataset" in os.listdir():
        shutil.rmtree("dataset")
    os.mkdir("dataset")
    np.save("dataset/signals.npy", signals)
    np.save("dataset/labels.npy",  labels)

else:
    signals=np.load("dataset/signals.npy")
    labels=np.load("dataset/labels.npy")


print("signals.shape:  ", signals.shape)
print("labels.shape:   ", labels.shape)




## load data
signals=np.load("dataset/signals.npy")
labels= np.load("dataset/labels.npy")

print(signals.shape)
print(labels.shape)




x=signals
y=labels

x_tv, x_test, y_tv, y_test = train_test_split(x, y, test_size=0.15, random_state=42) # tv means train and validation
x_train, x_val, y_train, y_val = train_test_split(x_tv, y_tv, test_size=0.1765, random_state=42)

mean=x_train.mean(axis=(0,2), keepdims=True)
std = x_train.std(axis=(0,2), keepdims=True)

x_train = (x_train -mean) / (std + 1e-8)
x_val   = (x_val   -mean) / (std + 1e-8)
x_test  = (x_test  -mean) / (std + 1e-8)

x_train = torch.tensor(x_train, dtype=torch.float)
x_val   = torch.tensor(x_val,   dtype=torch.float)
x_test  = torch.tensor(x_test,  dtype=torch.float)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val,   dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_dataset = TensorDataset(x_train, y_train)
val_dataset   = TensorDataset(x_val, y_val)
test_dataset  = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)




## Model Definition
class checkpoint_manager:
    def __init__(self, save_dir, loss_name="val_loss"):
        if(os.path.exists(save_dir)):
            shutil.rmtree(save_dir)        
        os.makedirs(save_dir)
        self.save_dir=save_dir
        self.loss_name=loss_name
        self.best_loss=float("inf")


    def is_better(self, loss):
        return loss < self.best_loss


    def save(self, model, optimizer, epoch, loss, scheduler=None, extra=None):
        state={
            "epoch": epoch, 
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            self.loss_name: loss,
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if extra is not None:
            state.update(extra)

        #save the last checkpoint:
        torch.save(state, os.path.join(self.save_dir, "last_checkpoint.pt"))

        #save the best checkpoint:
        if self.is_better(loss):
            self.best_loss=loss
            torch.save(state, os.path.join(self.save_dir, "best_checkpoint.pt"))
            print("new best checkpoint saved:\t", self.loss_name, ":  ", self.best_loss)



    def load(self, path, model, optimizer, scheduler=None, map_location="cpu"):
        checkpoint=torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint
        
        





class early_stopping:
    def __init__(self, patience=10):
        self.patience=patience
        self.counter=0
        self.best_loss=float("inf")
        self.should_stop=False

    def step(self, loss):
        improved = loss<self.best_loss
        if improved:
            self.best_loss=loss
            self.counter=0
        else:
            self.counter+=1
            if self.counter>=self.patience:
                self.should_stop=True
        return improved




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




model=Classifier().to(device)




def r2_loss(y_pred, y_true, eps=1e-8):
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=1)
    mean=torch.mean(y_true, dim=1, keepdim=True)
    ss_tot = torch.sum((y_true - mean) ** 2, dim=1)
    r2 = 1 - ss_res / (ss_tot + eps)
    return   (1 - r2).mean()


def custom_loss(output, y_batch, cls1_u, cls1_l, cls2_u, cls2_l):
    alpha = 0.2  #defined by the paper

    bce_loss=nn.BCEWithLogitsLoss()
    first_loss=bce_loss(output, y_batch.float())
    second_loss=( r2_loss(cls2_u, cls2_l) / r2_loss(cls1_u, cls1_l) )
    total_loss = first_loss + (alpha*second_loss)
    
    return total_loss    


optimizer=optim.AdamW(
    model.parameters(),
    lr=1e-3
)




epochs=200
if(os.path.exists("logs")):
    shutil.rmtree("logs")     
writer=SummaryWriter(log_dir="logs")
ckpt_manager=checkpoint_manager(save_dir="checkpoint_manager", loss_name="custom_paper's_loss")
early_stopper=early_stopping(patience=10)

for epoch in range(epochs):
    model.train()
    train_loss=0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        output, tokens = model(x)
        cls1_u, cls1_l, cls2_u, cls2_l = tokens
        
        loss=custom_loss(output, y, cls1_u, cls1_l, cls2_u, cls2_l)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_loader.dataset)

    #validation:
    model.eval()
    val_loss=0.0
    with torch.no_grad():
        for x, y in val_loader:
            x=x.to(device)
            y=y.to(device)
            output, tokens = model(x)
            cls1_u, cls1_l, cls2_u, cls2_l = tokens
            loss=custom_loss(output, y, cls1_u, cls1_l, cls2_u, cls2_l)
            val_loss += loss.item() * x.size(0)
    val_loss /= len(val_loader.dataset)
            
    #summary writer:
    current_lr=optimizer.param_groups[0]["lr"]
    print("\n", f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | "
          f"val_loss: {val_loss:.4f} | lr: {current_lr:.2e}")
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("LR", current_lr, epoch)
                
    #checkpoint saving:
    ckpt_manager.save(model, optimizer, epoch, val_loss, scheduler=None, extra=None)

    #early stopping:
    early_stopper.step(val_loss)
    if early_stopper.should_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
    else:
        print("\tEarly stopping did nothing.    early_stopper.should_stop:", early_stopper.should_stop)


writer.close()