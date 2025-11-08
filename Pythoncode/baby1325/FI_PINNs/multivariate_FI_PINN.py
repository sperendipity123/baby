import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker

import numpy as np
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import scipy.io
import scipy.stats as stats
from matplotlib.pyplot import MultipleLocator
# from timm.models.layers import trunc_normal_

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(3407)#3407
# Random number generators in other libraries
np.random.seed(3407)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(3407)


xl = -1  # x in [-1,1]
xr = 1
yl = -1  # y in [-1,1]
yr = 1

# Training  Data
class trainingdata(nn.Module):
    # B_p is boundary points and C_p is interior points
    def interior(C_p):
        # 内点
        # 数据采样对解有影响
        x = torch.rand(C_p, 1)    # 生成N*1 的0到1之间随机数
        x = (xr - xl) * x + xl
        y = torch.rand(C_p, 1)
        y = (yr - yl) * y + yl
        x=x.float().to(device)
        y=y.float().to(device)
        cond = -2000* torch.exp(-1000* ((x-0.5)**2+(y-0.5)**2))*(2000*(((x-0.5)**2+(y-0.5)**2))-2).float().to(device)
        # print(x)
        # print(y)
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_interior, y_interior, cond_interior = interior(C_p)
    # interior points need to be save
    def down(B_p):
        # 边界 u(x,-1)=e^(-10*(9/4+(x-0.5)^2))
        # 数据采样对解有影响
        x = torch.rand(B_p, 1)
        x = (xr - xl) * x + xl
        y = yl*torch.ones_like(x)
        x=x.float().to(device)
        y=y.float().to(device)
        cond = torch.exp(-1000*(9/4+(x-0.5)**2)).float().to(device)
        # cond = torch.exp(-10*(1/4+(x-0.5)**2))
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_down, y_down, cond_down = down(B_p)
    # down points need to be save
    def up(B_p):
        # 边界 u(x,1)=e^(-10*(1/4+(x-0.5)^2))
        # 数据采样对解有影响
        x = torch.rand(B_p, 1)
        x = (xr - xl) * x + xl
        y = yr*torch.ones_like(x)
        x=x.float().to(device)
        y=y.float().to(device)
        cond = torch.exp(-1000*(1/4+(x-0.5)**2)).float().to(device)
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_up, y_up, cond_up = up(B_p)
    # up points need to be save
    def left(B_p):
        # 边界 u(-1,y)=e^(-10*(9/4+(y-0.5)^2))
        # 数据采样对解有影响
        y = torch.rand(B_p, 1)
        y = (yr - yl) * y + yl
        x = - xl*torch.ones_like(y)
        x=x.float().to(device)
        y=y.float().to(device)
        cond = torch.exp(-1000*(9/4+(y-0.5)**2)).float().to(device)
        # cond = torch.exp(-10*(1/4+(y-0.5)**2))
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_left, y_left, cond_left = left(B_p)
    # left points need to be save
    
    def right(B_p):
        # 边界 u(1,y)=e^(-10*(1/4+(y-0.5)^2))
        y = torch.rand(B_p, 1)
        y = (yr - yl) * y + yl       # 数据采样对解有影响
        x = xr*torch.ones_like(y)
        x=x.float().to(device)
        y=y.float().to(device)
        cond = torch.exp(-1000*(1/4+(y-0.5)**2)).float().to(device)
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_right, y_right, cond_right = right(B_p)
    # right points need to be save

    def data_interior(C_p):
        # 内点
        # 数据采样对解有影响，数据误差是与真解进行loss计算
        x = torch.rand(C_p, 1)    # 生成N*1 的0到1之间随机数
        x = (xr - xl) * x + xl
        y = torch.rand(C_p, 1)
        y = (yr - yl) * y + yl
        x=x.float().to(device)
        y=y.float().to(device)
        cond = torch.exp(-1000 * ((x-0.5)**2+(y-0.5)**2)).float().to(device)
        return x.requires_grad_(True), y.requires_grad_(True), cond
     
    # x_data_interior, y_data_interior, cond_data_interior = data_interior(B_p)
    # data_interior points need to be save
    


ub = np.array([xr,yr])
lb = np.array([xl,yl])

def gradients(u, x, order=1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
        else:
             return gradients(gradients(u, x), x, order=order - 1)

# Physics Informed Neural Network

class PINNNet(nn.Module):

    def __init__(self,layers):
        super().__init__() #call __init__ from parent class 
              
        'activation function'
        # self.activation = nn.Tanh()
        self.activation = nn.Sigmoid()
        
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        self.iter = 0
        
        '''
        Alternatively:
        
        *all layers are callable 
    
        Simple linear Layers
        self.fc1 = nn.Linear(2,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,1)
        
        '''
    
        'Xavier Normal Initialization'
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers)-1):
            
            # weights from a normal distribution with 
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
    'foward pass'
    def forward(self,x):
        
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        
        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)
                   
        #preprocessing input 
        x = (x - l_b)/(u_b - l_b) #feature scaling
        
        #convert to float
        a = x.float()
                        
        '''     
        Alternatively:
        
        a = self.activation(self.fc1(a))
        a = self.activation(self.fc2(a))
        a = self.activation(self.fc3(a))
        a = self.fc4(a)
        
        '''
        
        for i in range(len(layers)-2):
            
            z = self.linears[i](a)
            # a = self.activation(z)
                        
            a = self.activation(z)*z
            
        a = self.linears[-1](a)
        return a
    
    # Loss function
# 以下6个损失是PDE损失

'loss function'
loss_function = nn.MSELoss(reduction ='mean')
       
def l_interior(x1,y1,cond1,net):
        # 损失函数L1
        # x_interior, y_interior, cond_interior = trainingdata.interior(C_p)
        
        uxy = torch.cat([x1, y1], dim=1).float().to(device)
        u = net(uxy).float().to(device)
        loss_interior = loss_function(- gradients(u, x1, 2)-gradients(u, y1, 2), cond1)
        return loss_interior
def l_abs(x1,y1,net):
        # 损失函数L1
        # x_interior, y_interior, cond_interior = trainingdata.interior(C_p)
        x1.requires_grad_(True)
        y1.requires_grad_(True)   # need add grad
        uxy = torch.cat([x1, y1], dim=1).float().to(device)
        u = net(uxy).float().to(device)
        cond1=-2000* torch.exp(-1000* ((x1-0.5)**2+(y1-0.5)**2))*(2000*(((x1-0.5)**2+(y1-0.5)**2))-2).float().to(device)
        loss_interior_abs=torch.abs(- gradients(u, x1, 2)-gradients(u, y1, 2)-cond1)
        return loss_interior_abs

def l_down(x2,y2,cond2,net):
        # 损失函数L2
        # x_down, y_down, cond_down = trainingdata.down(B_p)
        uxy = torch.cat([x2, y2], dim=1)
        u = net(uxy)
        loss_down = loss_function(u, cond2)
        return loss_down


def l_up(x3,y3,cond3,net):
        # 损失函数L3
        # x_up, y_up, cond_up = trainingdata.up(B_p)
        uxy = torch.cat([x3, y3], dim=1)
        u = net(uxy)
        loss_up = loss_function(u, cond3)
        return loss_up
    
def l_left(x4,y4,cond4,net):
        # 损失函数L4
        # x_left, y_left, cond_left = trainingdata.left(B_p)
        uxy = torch.cat([x4, y4], dim=1)
        u = net(uxy)
        loss_left = loss_function(u, cond4)
        return loss_left
    

def l_right(x5,y5,cond5,net):
        # 损失函数L5
        # x_right, y_right, cond_right = trainingdata.right(B_p)
        uxy = torch.cat([x5, y5], dim=1)
        u = net(uxy)
        loss_right = loss_function(u, cond5)
        return loss_right

    # 构造数据损失
def l_data(x6,y6,cond6,net):
        # 损失函数L6
        # x_data_interior, y_data_interior, cond_data_interior = trainingdata.data_interior(C_p)
        uxy = torch.cat([x6, y6], dim=1)
        u = net(uxy)
        loss_data_interior = loss_function(u, cond6)
        return loss_data_interior
    
def loss(x1,y1,cond1,x2,y2,cond2,x3,y3,cond3,x4,y4,cond4,x5,y5,cond5,x6,y6,cond6,net):
        # all the loss
        loss_interior= l_interior(x1,y1,cond1,net)
        loss_down = l_down(x2,y2,cond2,net)
        loss_up = l_up(x3,y3,cond3,net)
        loss_left = l_left(x4,y4,cond4,net)
        loss_right = l_right(x5,y5,cond5,net)
        loss_data_interior = l_data(x6,y6,cond6,net)
        loss_val = loss_interior+loss_down+loss_up+loss_left+loss_right+loss_data_interior 
        return loss_val

def multivariate_truncated_normal(mu,sigma,lower,upper,number):
    data=np.zeros([number,2])
    xunhuan=1
    while xunhuan <= number:
         x=stats.multivariate_normal.rvs(mean=mu,cov=sigma,size=1)
         x1=np.array(x)
         if (x1[0]>lower[0]) & (x1[0]<upper[0]) & (x1[1]>lower[1]) & (x1[1]<upper[1]):
              data[xunhuan-1,0]=x1[0]
              data[xunhuan-1,1]=x1[1]
              xunhuan=xunhuan+1
         else:
              xunhuan=xunhuan    
    data=torch.from_numpy(data).float().to(device)        
    return data
# 一个一个选点有尽头

# define self-adaptive importance sampling
def SAIS(N1, N2, p0, omega, M,net):
    #   N1 and N2 is the number of samples, p0 is a parameter
    #   g is LSF, omega is the prior distribution and M is the maximum updated number
    N1 = int(N1)  # 转化为整数
    N2 = int(N2)
    k = 1
    h_1 = omega

    while k <= M:
        # h_k = h[k - 1]
        if k == 1:
            # choose random N1 points for training
            miu_k = torch.tensor([[0,0]])
            Sigma_k = torch.tensor([[1,0],[0,1]])
             # 选出来的样本点，获得了N1个tensor样本点
            space_to_train= multivariate_truncated_normal(miu_k.cpu().numpy().ravel(),Sigma_k.cpu().numpy(),[xl,yl],[xr,yr],N1)
            u_shuzhi = net(space_to_train).float().to(device)
            uu = torch.exp(-1000*((space_to_train[:,0][:,None]-0.5)**2+(space_to_train[:,1][:,None]-0.5)**2)).float().to(device)
            loss_fc = torch.abs(u_shuzhi-uu) - epsilon_r
            # loss_interior_abs=l_abs(space_to_train[:,0][:,None],space_to_train[:,1][:,None],net)
            # loss_fc = loss_interior_abs - epsilon_r
            
        else:
            # 获得了N1个tensor样本点
            space_to_train= multivariate_truncated_normal(miu_k.cpu().numpy().ravel(),Sigma_k.cpu().numpy(),[xl,yl],[xr,yr],N1)
            u_shuzhi = net(space_to_train).float().to(device)
            uu = torch.exp(-1000*((space_to_train[:,0][:,None]-0.5)**2+(space_to_train[:,1][:,None]-0.5)**2)).float().to(device)
            loss_fc = torch.abs(u_shuzhi-uu) - epsilon_r
            # loss_interior_abs=l_abs(space_to_train[:,0][:,None],space_to_train[:,1][:,None],net)
            # loss_fc = loss_interior_abs - epsilon_r
        # loss_fc = torch.tensor(loss_fc)
        sorted1, indices1 = torch.sort(loss_fc, 0,descending=True)
        space_sorted = torch.zeros([indices1.shape[0], 2])
        for i in range(0, indices1.shape[0]):
            space_sorted[i, 0] = space_to_train[indices1[i][0], 0]
            space_sorted[i, 1] = space_to_train[indices1[i][0], 1]
        space_sorted = space_sorted.float().to(device)
        sum1 = 0
        for i in range(0, sorted1.shape[0]):
            if sorted1[i][0] > 0:
                sum1 = sum1 +1
        N_elta = sum1
        N_p = int(p0 * N_1)
        print(N_elta, N_p)
        if N_elta < N_p:
            miu_k1 = torch.zeros([1,1]).float().to(device)
            miu_k2 = torch.zeros([1,1]).float().to(device)
            for i in range(0, N_p):
                miu_k1 = miu_k1 + space_sorted[i, 0]
                miu_k2 = miu_k2 + space_sorted[i, 1]
            miu_kk = torch.cat([miu_k1 / N_p, miu_k2 / N_p],dim =1).float().to(device)
            sum1 = 0
            for i in range(0, N_p):
                xx =space_sorted[i, :] - miu_kk
                sum1 = sum1 + torch.kron(xx.reshape(-1,1), xx)
            Sigma_kk = sum1 / (N_p - 1)
            Sigma_kk=Sigma_kk.float().to(device)
            k = k + 1
            miu_k = miu_kk
            Sigma_k = Sigma_kk
        else:
            break
    miu_opt = miu_k
    Sigma_opt = Sigma_k
    # 获得了最优估计下的N2个样本点
    space_opt_train= multivariate_truncated_normal(miu_opt.cpu().numpy().ravel(),Sigma_opt.cpu().numpy(),[xl,yl],[xr,yr],N2)
    miu_opt=miu_opt.cpu().numpy()
    Sigma_opt=Sigma_opt.cpu().numpy()
    
    u_shuzhi = net(space_opt_train).float().to(device)
    uu = torch.exp(-1000*((space_opt_train[:,0][:,None]-0.5)**2+(space_opt_train[:,1][:,None]-0.5)**2)).float().to(device)
    loss_fc1 = torch.abs(u_shuzhi-uu) - epsilon_r
    # loss_interior_abs=l_abs(space_opt_train[:,0][:,None],space_opt_train[:,1][:,None],net)
    # loss_fc1 = loss_interior_abs - epsilon_r
    
    sorted2, indices2 = torch.sort(loss_fc1,0,descending=True)
    space_sorted = torch.zeros([indices2.shape[0], 2])
    sum2 = 0
    for i in range(0, sorted2.shape[0]):
        if sorted2[i][0] > 0:
            sum2 = sum2 +1
            space_sorted[sum2-1, 0] = space_opt_train[indices2[i][0], 0]
            space_sorted[sum2-1, 1] = space_opt_train[indices2[i][0], 1]
    D_adaptive = space_sorted[0:sum2,:].float().to(device)
    sum3=0
    for i in range(0,D_adaptive.shape[0]):
        
        h_opt=stats.multivariate_normal.pdf(D_adaptive[i,:].cpu().numpy(),mean=miu_opt.ravel(),cov=Sigma_opt)/(stats.multivariate_normal.cdf([xr,yr],mean=miu_opt.ravel(),cov=Sigma_opt)-stats.multivariate_normal.cdf([xl,yl],mean=miu_opt.ravel(),cov=Sigma_opt))
        sum3=sum3+1/h_opt
    P_f = sum3 / N2     
    return D_adaptive, P_f, Sigma_opt

####Solution Plot
def solutionplot(u_pred_fig,u_error_fig,u_real_fig):
    
    fig, ax = plt.subplots()
    fig.suptitle('Numerical results obtained by SAIS',fontsize = 12)
    ax.axis('off')
    
    gs = gridspec.GridSpec(2, 2)
    
    gs.update(top=1-0.07, bottom=0.1, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs[0, 0])

    h = ax.imshow((u_pred_fig).detach().cpu().numpy(), interpolation='nearest', cmap='rainbow', 
                extent=[xm.detach().cpu().numpy().min(), xm.detach().cpu().numpy().max(), ym.detach().cpu().numpy().min(), ym.detach().cpu().numpy().max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    
    # ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

    
    # ####### Row 1: u(t,x) slices ##################

    
    ax = plt.subplot(gs[0, 1])

    h = ax.imshow((u_error_fig).detach().cpu().numpy(), interpolation='nearest', cmap='rainbow', 
                extent=[xm.detach().cpu().numpy().min(), xm.detach().cpu().numpy().max(), ym.detach().cpu().numpy().min(), ym.detach().cpu().numpy().max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    
   
    ax = plt.subplot(gs[1, 0])
    ax.plot(xc.detach().cpu().numpy(),u_real_fig.detach().cpu().numpy().T[750,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(xc.detach().cpu().numpy(),u_pred_fig.detach().cpu().numpy().T[750,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(0.5,y)$')

    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([0,1.1]) 
    x_major_locator=MultipleLocator(0.5) #以每15显示
    y_major_locator=MultipleLocator(0.2)#以每3显示
    ax.axis('auto')
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)  
    # ax.set_title('$x = 0.5s$', fontsize = 8)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    ax.legend(prop = {'size':6})
    # fig.tight_layout()

    plt.savefig('multivariate_Possion PINN solve.jpg',dpi = 500)


""" def solutionplot(u_pred_fig,u_error_fig,u_real_fig):
    
    fig, ax = plt.subplots()
    fig.suptitle('Numerical results obtained by SAIS',fontsize = 12)
    ax.axis('off')
    # gs = fig.add_gridspec(2,2)
    plt.figure()
    ax=plt.gca()
    h = ax.imshow((u_pred_fig).detach().cpu().numpy(), interpolation='nearest', cmap='rainbow', 
                extent=[xm.detach().cpu().numpy().min(), xm.detach().cpu().numpy().max(), ym.detach().cpu().numpy().min(), ym.detach().cpu().numpy().max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    plt.savefig('Possion PINN SAIS solve.jpg')
    
    # ####### Row 1: u(t,x) slices ##################
    plt.figure()
    ax=plt.gca()
   
    h = ax.imshow((u_error_fig).detach().cpu().numpy(), interpolation='nearest', cmap='rainbow', 
                extent=[xm.detach().cpu().numpy().min(), xm.detach().cpu().numpy().max(), ym.detach().cpu().numpy().min(), ym.detach().cpu().numpy().max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    plt.savefig('Possion PINN SAIS abs error.jpg')

    
    plt.figure()
    ax=plt.gca()
   
    ax.plot(xc.detach().cpu().numpy(),u_real_fig.detach().cpu().numpy().T[750,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(xc.detach().cpu().numpy(),u_pred_fig.detach().cpu().numpy().T[750,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(0.5,y)$')

    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([0,1.1]) 
    x_major_locator=MultipleLocator(0.5) #以每15显示
    y_major_locator=MultipleLocator(0.2)#以每3显示
    ax.axis('auto')
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)  
    
    ax.legend(prop = {'size':6})
    

    plt.savefig('Possion PINN SAIS x at 0.5.jpg')  """





# Main
#  给网络输入参数
layers = np.array([2,50,50,50,20,20,20,20,1])#7 hidden layers   神经元个数会影响解的误差大小
net = PINNNet(layers) 
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001,betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)



#初始化 数据
epochs = 10000    # 训练代数
h = 1000    # 画图网格密度
C_p =2000  # 内点配置点数 5000
B_p = 200  # 边界点配置点数
M = 10
epsilon_p = 0.08
epsilon_r = 0.08
N_1 = 300
p_0 = 0.1
N_2 = 1000


x_interior, y_interior, cond_interior = trainingdata.interior(C_p)
x_down, y_down, cond_down = trainingdata.down(B_p)
x_up, y_up, cond_up = trainingdata.up(B_p)
x_left, y_left, cond_left = trainingdata.left(B_p)
x_right, y_right, cond_right = trainingdata.right(B_p)
x_data_interior, y_data_interior, cond_data_interior = trainingdata.data_interior(C_p)
start_time = time.time()
s=1
while s <= M:
    # 开始训练
    max_iter = epochs

    

    for i in range(max_iter):
        optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
        net.to(device)
        loss_1 = loss(x_interior, y_interior, cond_interior,x_down, y_down, cond_down,x_up, y_up, cond_up, x_left, y_left, cond_left,x_right, y_right, cond_right,x_data_interior, y_data_interior, cond_data_interior,net)

        loss_1.backward() #backprop

        optimizer.step()
        
        # if i % (max_iter/10) == 0:

            

        #     print(i)
    

    net.to(device)
    D_adaptive, P_f, Sigma_opt=SAIS(N_1, N_2, p_0, 1, M,net)
    D_adaptive.to(device)
    # print(D_adaptive)
    print(P_f)
    if P_f < epsilon_p:
        break
        
    else:
         x_interior = torch.cat([x_interior,D_adaptive[:,0][:,None]],dim=0).to(device)
         y_interior = torch.cat([y_interior,D_adaptive[:,1][:,None]],dim=0).to(device)
         x=D_adaptive[:,0][:,None]
         y=D_adaptive[:,1][:,None]
         cond1 = -2000* torch.exp(-1000 * ((x-0.5)**2+(y-0.5)**2))*(2000*(((x-0.5)**2+(y-0.5)**2))-2).float().to(device)
         cond_interior = torch.cat([cond_interior,cond1],dim=0).to(device)
         x_data_interior = torch.cat([x_data_interior,D_adaptive[:,0][:,None]],dim=0).to(device)
         y_data_interior = torch.cat([y_data_interior,D_adaptive[:,1][:,None]],dim=0).to(device)
         cond2 = torch.exp(-1000 * ((x-0.5)**2+(y-0.5)**2)).float().to(device)
         cond_data_interior = torch.cat([cond_data_interior,cond2],dim=0).to(device)
         s=s+1

# print(x_interior.shape)
print(s)
elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))

# Testing data
xc = torch.linspace(xl, xr, h)
yc = torch.linspace(yl, yr, h)
xm, ym = torch.meshgrid(xc, yc)
xx = xm.reshape(-1, 1)   #转成一列
yy = ym.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1).float().to(device)

# start test
u_real = torch.exp(-1000*((xx-0.5)**2 +(yy-0.5)**2)).float().to(device)
u_pred = net(xy)
u_error = (u_pred-u_real).float().to(device)

# plot
u_pred_fig = u_pred.reshape(h,h)
u_real_fig = u_real.reshape(h,h)
u_error_fig = u_error.reshape(h,h)
print("L_2 error is: ",float(torch.linalg.norm((u_real - u_pred), 2) / torch.linalg.norm(u_real,2)))

solutionplot(u_pred_fig,u_error_fig,u_real_fig)



