#define twisting tanh activation function
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

class TwistingTanh(torch.autograd.Function):
    '''
    this function is recommended for general purposes in deep learning 
    f(x) = 1.7159*tanh(2x/3)+ax; while a = 1
    '''
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return 1.7159 + torch.tanh(input * 2 / 3) + input
    
    @staticmethod
    def backward(ctx,grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (2.1438 - 1.1438 * (torch.tanh(2 * input / 3)) ** 2)
        return grad_input

#example for utilizing this activation function in pytorch	
dtype = torch.DoubleTensor
x = torch.from_numpy(np.linspace(-1,1,100))
y = x*2
x = torch.unsqueeze(x,dim=1)
y = torch.unsqueeze(y,dim=1)
x,y = Variable(x),Variable(y)
func = ScaleTanh.apply

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden_1,n_hidden_2,n_output):
        super(Net,self).__init__()
        self.hidden_1 = torch.nn.Linear(n_feature,n_hidden_1)
        self.hidden_2 = torch.nn.Linear(n_hidden_1,n_hidden_2)
        self.predict = torch.nn.Linear(n_hidden_2,n_output)
    def forward(self,x):
        x = func(self.hidden_1(x))
        x = func(self.hidden_2(x))
        x = self.predict(x)
        return x
    
net = Net(1,200,200,1)
net = net.double()
optimizer = torch.optim.SGD(net.parameters(),lr=0.001)
#stack flow error occuring in too small learning rate 
loss_func = torch.nn.MSELoss()

for t in range(500):    
    y_pred = net(x)
    loss = loss_func(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    print(t,loss)
    optimizer.step()
    if t % 5 == 0:
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),y_pred.data.numpy(),c='red')
        plt.show()