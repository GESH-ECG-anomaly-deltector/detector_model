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

from load_dataset import *
from CNN_blocks import *
from MSEL import *
from Transformer_Encoder import *
from TS_block import *
print("my codes are imported.\n")




device = "cpu"
print("device:", device)




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
    print("restart the program...")
    input()

else:
    signals=np.load("dataset/signals.npy")
    labels=np.load("dataset/labels.npy")


print("signals.shape:  ", signals.shape)
print("labels.shape:   ", labels.shape)

print("dataset is loaded.\n")




x=signals
y=labels

x_tv, x_test, y_tv, y_test = train_test_split(x, y, test_size=0.15, random_state=42) # tv means train and validation
x_train, x_val, y_train, y_val = train_test_split(x_tv, y_tv, test_size=0.1765, random_state=42)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)

mean=x_train.mean(axis=(0,2), keepdims=True)
std = x_train.std(axis=(0,2), keepdims=True)

np.save("train_mean.npy", mean)
np.save("train_std.npy", std)

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

print("standard scalling is performed.")
print("dataset splittion is done. train, validation and test DataLoaders are created.\n")




## Model Definition
class checkpoint_manager:
    def __init__(self, save_dir, loss_name="val_loss", save_mode=True):
        if(save_mode):
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
        

print("checkpoint_manager() is defined.\n")




class early_stopping:
    def __init__(self, patience=100):
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

print("early_stopping() is defined.\n")




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
        classification_output  =  torch.sigmoid( self.network(net_in) )
        
        return classification_output, [upper_cls1, lower_cls1, upper_cls2, lower_cls2]


print("classifier() is defined.\n")



device="cuda"
print("device: ", device)
model=Classifier().to(device)

def r2_loss(y_pred, y_true, eps=1e-8):
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=1)
    mean=torch.mean(y_true, dim=1, keepdim=True)
    ss_tot = torch.sum((y_true - mean) ** 2, dim=1)
    r2 = 1 - ss_res / (ss_tot + eps)
    return   (1 - r2).mean()


def custom_loss(output, y_batch, cls1_u, cls1_l, cls2_u, cls2_l):
    alpha = 0.2  #defined by the paper

    bce_loss=nn.BCEWithLogitsLoss() # this function will apply sigmoid() itself, se we use torch.logit()  to make inverse of sigmoid.
    first_loss=bce_loss(torch.logit(output), y_batch.float())
    second_loss=( r2_loss(cls2_u, cls2_l) / r2_loss(cls1_u, cls1_l) )
    total_loss = first_loss + (alpha*second_loss)
    
    return total_loss    


optimizer=optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=200,
    eta_min=1e-6
)


print("model is gone to device.")
print("costum_loss, optimizer and scheduler are implemented.\n")




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




epochs=200
if(os.path.exists("training_logs")):
    shutil.rmtree("training_logs")     
writer=SummaryWriter(log_dir="training_logs")
ckpt_manager=checkpoint_manager(save_dir="checkpoint_manager", loss_name="custom_paper's_loss")
early_stopper=early_stopping(patience=100)

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

            probs = torch.sigmoid(output)   # do this explicitly, don't rely on auto-sigmoid

            f1_micro.update(probs, y)
            f1_macro.update(probs, y)
            f1_per_label.update(probs, y)
            recall_micro.update(probs, y)
            recall_macro.update(probs, y)
            recall_per_label.update(probs, y)
            auroc_per_label.update(probs, y)
            acc_per_label.update(probs, y)
            acc_micro.update(probs, y)
            acc_macro.update(probs, y)
            
    val_loss /= len(val_loader.dataset)
            
    #summary writer:
    current_lr=optimizer.param_groups[0]["lr"]
    print("\n", f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | "
          f"val_loss: {val_loss:.4f} | lr: {current_lr:.2e}")
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("LR", current_lr, epoch)

    f1_micro_val = f1_micro.compute().item()
    f1_macro_val = f1_macro.compute().item()
    f1_per_label_val = f1_per_label.compute()          # tensor of shape [num_labels]
    recall_micro_val = recall_micro.compute().item()
    recall_macro_val = recall_macro.compute().item()
    recall_per_label_val = recall_per_label.compute()
    auroc_per_label_val = auroc_per_label.compute()
    acc_per_label_val = acc_per_label.compute()
    acc_micro_val = acc_micro.compute().item()
    acc_macro_val = acc_macro.compute().item()
    # log scalars
    writer.add_scalar("F1/micro", f1_micro_val, epoch)
    writer.add_scalar("F1/macro", f1_macro_val, epoch)
    writer.add_scalar("Recall/micro", recall_micro_val, epoch)
    writer.add_scalar("Recall/macro", recall_macro_val, epoch)
    writer.add_scalar("Acc/micro", acc_micro_val, epoch)
    writer.add_scalar("Acc/macro", acc_macro_val, epoch)
    # log per-label tensors, one scalar per label
    for i in range(num_labels):
        writer.add_scalar(f"F1_per_label/label_{i}", f1_per_label_val[i].item(), epoch)
        writer.add_scalar(f"Recall_per_label/label_{i}", recall_per_label_val[i].item(), epoch)
        writer.add_scalar(f"AUROC_per_label/label_{i}", auroc_per_label_val[i].item(), epoch)
        writer.add_scalar(f"Acc_per_label/label_{i}", acc_per_label_val[i].item(), epoch)
    # reset for next epoch — otherwise state accumulates across epochs!
    for m in [f1_micro, f1_macro, f1_per_label, recall_micro, recall_macro,
              recall_per_label, auroc_per_label, acc_per_label, acc_micro, acc_macro]:
        m.reset()

        
    #checkpoint saving:
    metrics_extra = {
        "f1_micro": f1_micro_val,
        "f1_macro": f1_macro_val,
        "f1_per_label": f1_per_label_val,#.cpu(),
        "recall_micro": recall_micro_val,
        "recall_macro": recall_macro_val,
        "recall_per_label": recall_per_label_val,#.cpu(),
        "auroc_per_label": auroc_per_label_val,#.cpu(),
        "acc_per_label": acc_per_label_val,#.cpu(),
        "acc_micro": acc_micro_val,
        "acc_macro": acc_macro_val,
    }
    ckpt_manager.save(model, optimizer, epoch, val_loss, scheduler=scheduler, extra=metrics_extra)


    #early stopping:
    early_stopper.step(val_loss)
    if early_stopper.should_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
    else:
        print("\tEarly stopping did nothing.    early_stopper.should_stop:", early_stopper.should_stop)

    scheduler.step()


writer.close()




print("\n\n*** *** *** *** *** ***")
print("end of training.\n")