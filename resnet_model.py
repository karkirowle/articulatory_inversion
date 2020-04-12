
import torch.nn as nn
import torch.nn.functional as F


class DBLSTM2(nn.Module):
    def __init__(self, args):
        super(DBLSTM2, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.input_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size,args.hidden_size)

        block_list = [args.hidden_size,32,64,64,64]

        # Max block num here: 4

        resnet_block = []

        for i in range(1,args.block_num):
            resnet_block.append(ResNetBlock(block_list[i-1],block_list[i]))
            resnet_block.append(ResNetBlock(block_list[i],block_list[i]))


        self.resnet_blocks = nn.Sequential(*resnet_block)


        self.fc_after = nn.Linear(64,args.hidden_size)
        self.num_layers = 2
        self.lstm1 = nn.LSTM(args.hidden_size,args.hidden_size_2,
                             bidirectional=True,num_layers=self.num_layers,batch_first=True)
        self.fc3 = nn.Linear(args.hidden_size_2 * 2,args.num_classes)

    def forward(self, input):

        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        residual = out
        out = out.view(out.shape[0],out.shape[2],out.shape[1],1)
        out = self.resnet_blocks(out)
        # out = F.relu(self.bn1(out))
        # out = self.res1(out)
        # out = self.res2(out)
        # out = self.conv2(out)
        # out = F.relu(self.bn2(out))
        # out = self.res3(out)
        # out = self.res4(out)

        out = self.fc_after(out.view(out.shape[0],out.shape[2],out.shape[1]))

        #out = self.fc_after_resnet(out)

        # Skip connection introduced so we must reach baseline
        out = residual + out

        #out = out.view(out.shape[0],out.shape[2],out.shape[1])
        out,hidden = self.lstm1(out)

        out = self.fc3(out)

        return out


class ResNetBlock(nn.Module):
    def __init__(self, in_filter, out_filter):
        super(ResNetBlock,self).__init__()

        self.conv0 = nn.Conv2d(in_filter, out_filter, kernel_size=(1,1), padding=(0,0))
        self.bn0 = nn.BatchNorm2d(out_filter)
        self.conv1 = nn.Conv2d(in_filter,out_filter,kernel_size=(3,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(out_filter)
        self.conv2 = nn.Conv2d(out_filter,out_filter,kernel_size=(3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(out_filter)

    def forward(self, input):


        out = F.relu(self.bn1(self.conv1(input)))
        out = self.bn0(self.conv0(input)) + self.bn2(self.conv2(out))
        return out