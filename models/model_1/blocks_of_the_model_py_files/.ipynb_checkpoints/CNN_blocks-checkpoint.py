## This code is Checked by chatgpt




import torch.nn as nn
import torch




class conv_block(nn.Module):
    def __init__(self, filters, stride, padding, out_channels, in_channels):
        super().__init__()

        self.block=nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=filters,
                stride=stride,
                padding=padding                
            )
        )


    def forward(self, x):
        return self.block(x)




class max_pool(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool=nn.MaxPool1d(
            kernel_size=1,
            stride=1,
            padding=0
        )


    def forward(self, x):
        return self.pool(x)




class batch_norm(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.bn_block=nn.Sequential(
            nn.BatchNorm1d(num_features)
        )


    def forward(self, x):
        return self.bn_block(x)




class relu_block(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu=nn.ReLU()


    def forward(self, x):
        return self.relu(x)




class CNN_blocks(nn.Module):
    def __init__(self):
        super().__init__()

        conv_layers_info=[
            #filters    stride    padding    out_channels    in_channels
             [ 7,         1,        3,        128,             12   ],     #conv1
             [ 5,         1,        2,        256,             128  ],     #conv2
             [ 3,         1,        1,        128,             256  ],     #conv3
             [ 1,         1,        0,        32,              128  ],     #conv4
             [ 1,         1,        0,        32 ,             128  ],     #conv5   
             [ 39,        1,        19,       32,              32   ],     #conv6
             [ 19,        1,        9,        32,              32   ],     #conv7
             [ 9 ,        1,        4,        32,              32   ],     #conv8
             [ 1 ,        1,        0,        128,             128  ],     #conv9
        ]

        self.conv_blocks=nn.ModuleList()
        self.conv_blocks.append(None)
        for i in range(len(conv_layers_info)):
            block=conv_block(
                filters      = conv_layers_info[i][0],
                stride       = conv_layers_info[i][1],
                padding      = conv_layers_info[i][2],
                out_channels = conv_layers_info[i][3],
                in_channels  = conv_layers_info[i][4]
            )
            self.conv_blocks.append(block)
        
        self.max_pool=max_pool()

        self.batch_norm1=batch_norm(128)

        #self.concat_batch_norm=>  it is defined in forward()
        self.batch_norm2=batch_norm(32*4)

        #element-wise addition is defined in forward()

        self.relu=relu_block()



    def forward(self, x0):
        #first element of the list is None.
        x1=self.conv_blocks[1](x0)
        x2=self.conv_blocks[2](x1)
        x3=self.conv_blocks[3](x2)

        x_maxpool=self.max_pool(x3)
        x4=self.conv_blocks[4](x3)
        x9=self.conv_blocks[9](x3)

        x5=self.conv_blocks[5](x_maxpool)
        x6=self.conv_blocks[6](x4)
        x7=self.conv_blocks[7](x4)
        x8=self.conv_blocks[8](x4)

        x_bn1=self.batch_norm1(x9)
        
        #concat:
        xc_5678=torch.cat([x5, x6, x7, x8], dim=1)
        #element-wise addition:
        x_bn2=self.batch_norm2(xc_5678)
        x_bn = x_bn1 + x_bn2

        x_relu=self.relu(x_bn)

        return x_relu



#128 channels with 1000 samples will give as output



