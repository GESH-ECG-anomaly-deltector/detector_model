## load libraries
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from model_blocks.CNN_blocks import *
from model_blocks.MSEL import *
from model_blocks.Transformer_Encoder import *
from model_blocks.TS_block import *




device = "cpu"


## load data:
x_test=np.load("samples_dataset/samples/sample_122.npy")
x_test_to_graph=x_test
mean=np.load("samples_dataset/train_mean.npy")
std= np.load("samples_dataset/train_std.npy")
x_test  = (x_test  -mean) / (std + 1e-8)
x_test  = torch.tensor(x_test,  dtype=torch.float)




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
        
model=Classifier().to(device)




print("classification output:")
checkpoint = torch.load(
    "model_2_trained_on_server/checkpoint_manager/best_checkpoint.pt",
    map_location=device
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
outputs=[]
with torch.no_grad():
        outputs.append(model(x_test.to(device))[0])
output= (torch.cat(outputs), None)
y_test_pred=torch.sigmoid(output[0])
y_test_pred=(y_test_pred>0.5).int()
print(y_test_pred)

x_test=x_test_to_graph
#save plot:
sample_rate=100
time=np.arange(x_test.shape[1])/sample_rate
leads=["I",  "II", "III", "aVR", "aVL", "aVF", 
       "V1", "V2", "V3",  "V4",  "V5",  "V6"]

fig, axes=plt.subplots(12, 1, figsize=(14, 18), sharex=True)
for i, ax in enumerate(axes):
     ax.plot(time, x_test[i])
     ax.set_ylabel(leads[i])
     ax.grid()

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("ecg_plot.png", dpi=300, bbox_inches="tight")