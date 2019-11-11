import torch.nn as nn
import torch as t
import torch.optim as optim # 引入优化方法
from torch.autograd import Variable
import torch.nn.functional as F #激活函数
#继承nn.Module，网络的封装
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #input:(1,1,32,32)
        #in_channel,out_channel,kernel_size,stride,padding
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2=nn.Conv2d(6,16,5)
        #y=wx+b
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        #卷积->激活->池化，重复两次
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        #x = x.view(batchsize, -1)中batchsize指转换后有几行，而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
net=CNN()
#print(net)
#print(list(net.parameters())) # 输出网络可学习的参数
input=Variable(t.randn(1,1,32,32))
target=Variable(t.arange(0,10,dtype=t.float32).view(1,10)) # 真实值
criterion=nn.MSELoss() # 均方误差
optimizer=optim.SGD(net.parameters(),lr=0.01) # 新建优化器，指定要调整的参数和学习率
optimizer.zero_grad() # 在训练过程开始前先梯度清零，和net.zero_grad()效果一样
output=net(input)
loss=criterion(output,target) # 计算损失
loss.backward() # 反向传播
optimizer.step() #更新参数
