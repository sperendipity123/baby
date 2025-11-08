import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.autograd
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from torch.optim import lr_scheduler
import numpy as np
import time
from pyDOE import lhs         #Latin Hypercube Sampling
from scipy.io import savemat
import scipy.stats as stats
from matplotlib.pyplot import MultipleLocator
from scipy.stats import qmc
import scipy.io
# from timm.models.layers import trunc_normal_

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(3407)#3407
# Random number generators in other libraries
np.random.seed(3407)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda:0': 
    print(torch.cuda.get_device_name()) 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(3407)


xl = 0  # x in [-1,1]
xr = 1
yl = 0  # y in [-1,1]
yr = 1

# Data Prep
data_finte = scipy.io.loadmat('/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/data/example2.mat')  # note change
D=data_finte['D'].astype(float)
E=data_finte['E'].astype(float)
u0=data_finte['u0'].astype(float)
bhgs=data_finte['bhgs']
bjd=data_finte['bjd'].astype(float)
Dbjd=data_finte['Dbjd'].astype(float)
space=data_finte['space']
space=space[0][0]



# crete high approximate u solution
def choose_traingle(x,y,u0,space):
    
    # define np function which means input and output are all np type
    for xunhuan in range(1,space+1):
        i=int(D[xunhuan-1,0])
        j=int(D[xunhuan-1,1])
        m=int(D[xunhuan-1,2])
        xi=E[i-1,0]
        xj=E[j-1,0]
        xm=E[m-1,0]
        yi=E[i-1,1]
        yj=E[j-1,1]
        ym=E[m-1,1]
        a1=np.mat([[yj,1],[ym,1]])
        ai=np.linalg.det(a1)
        a2=np.mat([[ym,1],[yi,1]])
        aj=np.linalg.det(a2)
        a3=np.mat([[yi,1],[yj,1]])
        am=np.linalg.det(a3)
        b1=np.mat([[xj,1],[xm,1]])
        bi=-np.linalg.det(b1)
        b2=np.mat([[xm,1],[xi,1]])
        bj=-np.linalg.det(b2)
        b3=np.mat([[xi,1],[xj,1]])
        bm=-np.linalg.det(b3)
        c1=np.mat([[xj,yj],[xm,ym]])
        ci=np.linalg.det(c1)
        c2=np.mat([[xm,ym],[xi,yi]])
        cj=np.linalg.det(c2)
        c3=np.mat([[xi,yi],[xj,yj]])
        cm=np.linalg.det(c3)
        d=np.mat([[xi,yi,1],[xj,yj,1],[xm,ym,1]])
        e=0.5*np.linalg.det(d)
        
        Ni=(ai*x+bi*y+ci)/(2*e)
        Nj=(aj*x+bj*y+cj)/(2*e)
        Nm=(am*x+bm*y+cm)/(2*e)
        pij=np.mat([[xj-xi,yj-yi]])
        pjm=np.mat([[xm-xj,ym-yj]])
        pmi=np.mat([[xi-xm,yi-ym]])
        pix1=np.mat([[x-xi,y-yi]])
        pjx1=np.mat([[x-xj,y-yj]])
        pmx1=np.mat([[x-xm,y-ym]])
        if (pij[0,0]*pix1[0,1]-pij[0,1]*pix1[0,0]>=0) and (pjm[0,0]*pjx1[0,1]-pjm[0,1]*pjx1[0,0]>=0) and (pmi[0,0]*pmx1[0,1]-pmi[0,1]*pmx1[0,0]>=0):
                uh=u0[i-1,0]*Ni+u0[j-1,0]*Nj+u0[m-1,0]*Nm
                uhx=(ai*u0[i-1,0]+aj*u0[j-1,0]+am*u0[m-1,0])/(2*e)
                uhy=(bi*u0[i-1,0]+bj*u0[j-1,0]+bm*u0[m-1,0])/(2*e)
    # u=uh.float().to(device)
    return uh,uhx,uhy
def high_approximate(x,y,u0,space):
    # define tensor function which means input and output are all tensor type
    uh=torch.zeros(x.shape[0],1).cpu().detach().float().to(device)
    uhx=torch.zeros(x.shape[0],1).cpu().detach().float().to(device)
    uhy=torch.zeros(x.shape[0],1).cpu().detach().float().to(device)
    for idx in range(0,x.shape[0]):
         uh[idx][0],uhx[idx][0],uhy[idx][0]=choose_traingle(x[idx][0].cpu().item(),y[idx][0].cpu().item(),u0,space)
    u=uh.float().to(device)
    ux=uhx.float().to(device)
    uy=uhy.float().to(device)
    return u,ux,uy

# use u=high_approximate(xxx[:,0][:,None],xxx[:,1][:,None],u0,space)

# 定义截断高斯抽样
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
# define regualation function
def psi_t(t,epsilon):
     return t/torch.sqrt(t**2+epsilon**2)
# define sobol squeece
def generate_sobol(n):
     sampler = qmc.Sobol(d=2, scramble=False, seed=10,optimization='random-cd')  # 维度
     sample = sampler.random(n)
     sample = qmc.scale(sample, l_bounds=[xl,yl], u_bounds=[xr,yr])
     return torch.from_numpy(sample).float().to(device) 


# Training  Data
class trainingdata(nn.Module):
    # B_p is boundary points and C_p is interior points
    def interior(C_p):
        # 内点
        # 数据采样对解有影响
        # x=data_interior[:,0][:,None]
        # y=data_interior[:,1][:,None]
        x = torch.rand(C_p, 1)    # 生成N*1 的0到1之间随机数
        x = (xr - xl) * x + xl
        y = torch.rand(C_p, 1)
        y = (yr - yl) * y + yl
        x=x.float().to(device)
        y=y.float().to(device)
        cond = ((50*x**4-100*x**3-2937.5*x**2+2987.5*x-450)*(x-0.5)**2+(50*y**4-100*y**3-2937.5*y**2+2987.5*y-450)*(y-0.5)**2).float().to(device)
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
        # x=torch.linspace(xl, xr, B_p).reshape(B_p,1)
        y = yl*torch.ones_like(x)
        x=x.float().to(device)
        y=y.float().to(device)
        cond = torch.zeros(B_p,1).float().to(device)
        # cond = (-6.25+g*(100*(x*(x-1)*(x-0.5)**4))/torch.sqrt((100*(x*(x-1)*(x-0.5)**4))**2+epsilon**2)).float().to(device)
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_down, y_down, cond_down = down(B_p)
    # down points need to be save
    def up(B_p):
        # 边界 u(x,1)=e^(-10*(1/4+(x-0.5)^2))
        # 数据采样对解有影响
        x = torch.rand(B_p, 1)
        x = (xr - xl) * x + xl
        # x=torch.linspace(xl, xr, B_p).reshape(B_p,1)
        y = yr*torch.ones_like(x)
        x=x.float().to(device)
        y=y.float().to(device)
        cond = torch.zeros(B_p,1).float().to(device)
        # cond=(6.25+g*(100*(x*(x-1)*(x-0.5)**4))/torch.sqrt((100*(x*(x-1)*(x-0.5)**4))**2+epsilon**2)).float().to(device)
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_up, y_up, cond_up = up(B_p)
    # up points need to be save
    def left(B_p):
        # 边界 u(-1,y)=e^(-10*(9/4+(y-0.5)^2))
        # 数据采样对解有影响
        y = torch.rand(B_p, 1)
        y = (yr - yl) * y + yl
        # y=torch.linspace(yl, yr, B_p).reshape(B_p,1)
        x = xl*torch.ones_like(y)
        x=x.float().to(device)
        y=y.float().to(device)
        cond = torch.zeros(B_p,1).float().to(device)
        # cond=(-6.25+g*(100*(y*(y-1)*(y-0.5)**4))/torch.sqrt((100*(y*(y-1)*(y-0.5)**4))**2+epsilon**2)).float().to(device)
        # cond = torch.exp(-10*(1/4+(y-0.5)**2))
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_left, y_left, cond_left = left(B_p)
    # left points need to be save
    
    def right(B_p):
        # 边界 u(1,y)=e^(-10*(1/4+(y-0.5)^2))
        y = torch.rand(B_p, 1)
        y = (yr - yl) * y + yl     # 数据采样对解有影响
        # y=torch.linspace(yl, yr, B_p).reshape(B_p,1)
        x = xr*torch.ones_like(y)
        x=x.float().to(device)
        y=y.float().to(device)
        cond =  torch.zeros(B_p,1).float().to(device)
        # cond=(6.25+g*(100*(y*(y-1)*(y-0.5)**4))/torch.sqrt((100*(y*(y-1)*(y-0.5)**4))**2+epsilon**2)).float().to(device)
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_right, y_right, cond_right = right(B_p)
    # right points need to be save
""" 
    def data_interior(C_p):
        # 内点
        # 数据采样对解有影响,数据误差是与真解进行loss计算
        x=data_interior[:,0][:,None]
        y=data_interior[:,1][:,None]
        x=x.float().to(device)
        y=y.float().to(device)
        cond = (100*(x*(x-1)*(x-0.5)**4+y*(y-1)*(y-0.5)**4)).float().to(device)
        return x.requires_grad_(True), y.requires_grad_(True), cond """
     
    # x_data_interior, y_data_interior, cond_data_interior = data_interior(B_p)
    # data_interior points need to be save

# 矫正网络的训练点
# Training  Data
class trainingdata1(nn.Module):
    # B_p is boundary points and C_p is interior points
    def interior1(C_p):
        # 内点
        # 数据采样对解有影响
        # x=data_interior[:,0][:,None]
        # y=data_interior[:,1][:,None]
        x = torch.rand(C_p, 1)    # 生成N*1 的0到1之间随机数
        x = (xr - xl) * x + xl
        y = torch.rand(C_p, 1)
        y = (yr - yl) * y + yl
        x=x.float().to(device)
        y=y.float().to(device)
        cond =torch.zeros(C_p,1).float().to(device)
        # print(x)
        # print(y)
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_interior, y_interior, cond_interior = interior(C_p)
    # interior points need to be save
    def down1(B_p):
        # 边界 u(x,-1)=e^(-10*(9/4+(x-0.5)^2))
        # 数据采样对解有影响
        x = torch.rand(B_p, 1)
        x = (xr - xl) * x + xl
        # x=torch.linspace(xl, xr, B_p).reshape(B_p,1)
        y = yl*torch.ones_like(x)
        x=x.float().to(device)
        y=y.float().to(device)
        cond = torch.zeros(B_p,1).float().to(device)
        # cond = (-6.25+g*(100*(x*(x-1)*(x-0.5)**4))/torch.sqrt((100*(x*(x-1)*(x-0.5)**4))**2+epsilon**2)).float().to(device)
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_down, y_down, cond_down = down(B_p)
    # down points need to be save
    def up1(B_p):
        # 边界 u(x,1)=e^(-10*(1/4+(x-0.5)^2))
        # 数据采样对解有影响
        x = torch.rand(B_p, 1)
        x = (xr - xl) * x + xl
        # x=torch.linspace(xl, xr, B_p).reshape(B_p,1)
        y = yr*torch.ones_like(x)
        x=x.float().to(device)
        y=y.float().to(device)
        cond = torch.zeros(B_p,1).float().to(device)
        # cond=(6.25+g*(100*(x*(x-1)*(x-0.5)**4))/torch.sqrt((100*(x*(x-1)*(x-0.5)**4))**2+epsilon**2)).float().to(device)
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_up, y_up, cond_up = up(B_p)
    # up points need to be save
    def left1(B_p):
        # 边界 u(-1,y)=e^(-10*(9/4+(y-0.5)^2))
        # 数据采样对解有影响
        y = torch.rand(B_p, 1)
        y = (yr - yl) * y + yl
        # y=torch.linspace(yl, yr, B_p).reshape(B_p,1)
        x = xl*torch.ones_like(y)
        x=x.float().to(device)
        y=y.float().to(device)
        cond = torch.zeros(B_p,1).float().to(device)
        # cond=(-6.25+g*(100*(y*(y-1)*(y-0.5)**4))/torch.sqrt((100*(y*(y-1)*(y-0.5)**4))**2+epsilon**2)).float().to(device)
        # cond = torch.exp(-10*(1/4+(y-0.5)**2))
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_left, y_left, cond_left = left(B_p)
    # left points need to be save
    
    def right1(B_p):
        # 边界 u(1,y)=e^(-10*(1/4+(y-0.5)^2))
        y = torch.rand(B_p, 1)
        y = (yr - yl) * y + yl     # 数据采样对解有影响
        # y=torch.linspace(yl, yr, B_p).reshape(B_p,1)
        x = xr*torch.ones_like(y)
        x=x.float().to(device)
        y=y.float().to(device)
        cond =  torch.zeros(B_p,1).float().to(device)
        # cond=(6.25+g*(100*(y*(y-1)*(y-0.5)**4))/torch.sqrt((100*(y*(y-1)*(y-0.5)**4))**2+epsilon**2)).float().to(device)
        return x.requires_grad_(True), y.requires_grad_(True), cond
    
    # x_right, y_right, cond_right = right(B_p)
    # right points need to be save



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
        self.activation = nn.Tanh()
        
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
                        
            a = self.activation(z)
            
        a = self.linears[-1](a)
        return a
    
# 矫正的网络结构    
class Newnet(nn.Module):

    def __init__(self,layers2):
        super().__init__() #call __init__ from parent class 
              
        'activation function'
        # self.activation = nn.Tanh()
        self.activation = nn.Tanh()
        
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers2[i], layers2[i+1]) for i in range(len(layers2)-1)])
        
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
        for i in range(len(layers2)-1):
            
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
        
        for i in range(len(layers2)-2):
            
            z = self.linears[i](a)
            # a = self.activation(z)
                        
            a = self.activation(z)
            
        a = self.linears[-1](a)
        return a       
    
    
# Loss function
# 以下6个损失是PDE损失

'loss function'

loss_function = nn.MSELoss(reduction ='mean')
# Compute the empirical NTK = J J^T
"""  L=list(net.parameters())
            
    Ker = torch.zeros((x1.shape[0],x1.shape[0])).float().to(device)
    for k in range(len(L)):
        Jacobi=torch.zeros((x1.shape[0],L[k].shape[0]*L[k].shape[1])).float().to(device)
        for i in range(x1.shape[0]):
            J=(gradients((- gradients(u, x1, 2)-gradients(u, y1, 2)+alpha*u)[i,0],L[k],1))
            J1 = torch.reshape(J, shape=(L[k].shape[0]*L[k].shape[1],-1)).T
            Jacobi[i,:]=J1.float().to(device)
        K=Jacobi@Jacobi.T
        Ker=Ker+K
    Kuu=compute_ntk(J,uxy,J,uxy) """
# 计算pinn的损失函数
class pdeloss1(nn.Module):
    def l_interior(x1,y1,cond1,net):
            # 损失函数L1
            uxy = torch.cat([x1, y1], dim=1)
            u = net(uxy).float().to(device)
        
            loss_interior = loss_function(- gradients(u, x1, 2)-gradients(u, y1, 2)+alpha*u, cond1)
            
            return loss_interior

    def l_down(x2,y2,cond2,net):
            # 损失函数L2
            # x_down, y_down, cond_down = trainingdata.down(B_p)
            # data_gen=TensorDataset(x1,y1)
            # dataset = DataLoader(dataset = data_gen,
            #              batch_size = 32,
            #              shuffle=True)
            # u = net(dataset).float().to(device)
            uxy = torch.cat([x2, y2], dim=1)
            u = net(uxy).float().to(device)
            loss_down = loss_function(-gradients(u,y2,1)+g*psi_t(u,epsilon), cond2)
            return loss_down

    def l_up(x3,y3,cond3,net):
            # 损失函数L3
            # x_up, y_up, cond_up = trainingdata.up(B_p) psi_t(u,epsilon)
            uxy = torch.cat([x3, y3], dim=1)
            u = net(uxy).float().to(device)
            loss_up = loss_function(gradients(u,y3,1)+g*psi_t(u,epsilon), cond3)
            return loss_up
        
    def l_left(x4,y4,cond4,net):
            # 损失函数L4
            # x_left, y_left, cond_left = trainingdata.left(B_p)
            uxy = torch.cat([x4, y4], dim=1)
            u = net(uxy).float().to(device)
            loss_left = loss_function(-gradients(u,x4,1)+g*psi_t(u,epsilon), cond4)
            return loss_left
        

    def l_right(x5,y5,cond5,net):
            # 损失函数L5
            # x_right, y_right, cond_right = trainingdata.right(B_p)
            uxy = torch.cat([x5, y5], dim=1)
            u = net(uxy).float().to(device)
            loss_right = loss_function(gradients(u,x5,1)+g*psi_t(u,epsilon), cond5)
            return loss_right
        
    def loss(x1,y1,cond1,x2,y2,cond2,x3,y3,cond3,x4,y4,cond4,x5,y5,cond5,net):
            # all the loss loss(x1,y1,cond1,x2,y2,cond2,x3,y3,cond3,x4,y4,cond4,x5,y5,cond5,x6,y6,cond6,net):
            loss_interior= pdeloss1.l_interior(x1,y1,cond1,net)
            loss_down = pdeloss1.l_down(x2,y2,cond2,net)
            loss_up = pdeloss1.l_up(x3,y3,cond3,net)
            loss_left = pdeloss1.l_left(x4,y4,cond4,net)
            loss_right = pdeloss1.l_right(x5,y5,cond5,net)
            """ loss_data_interior = l_data(x6,y6,cond6,net)
            lossf_qrad=f_qrad(x1,y1,net) """
            loss_val = 1*(loss_interior)+1*(loss_down+loss_up+loss_left+loss_right)
            """ +10*lossf_qrad +loss_data_interior"""
            # +10**(-4)*loss_grad
            return loss_val

    def closure():
                
                    
        loss_1 = pdeloss1.loss(x_interior, y_interior, cond_interior,x_down, y_down, cond_down,x_up, y_up, cond_up, x_left, y_left, cond_left,x_right, y_right, cond_right,net)     
        optimizer.zero_grad()
        
        # loss_1.backward()  loss_1 = loss(x_interior, y_interior, cond_interior,x_down, y_down, cond_down,x_up, y_up, cond_up, x_left, y_left, cond_left,x_right, y_right, cond_right,x_data_interior, y_data_interior, cond_data_interior,net)  
        loss_1.backward(retain_graph=True)
        
        return loss_1

class pdeloss2(nn.Module):
    def l_interior1(x1,y1,cond1,Rnet,net):
            # 损失函数L1
            uxy = torch.cat([x1, y1], dim=1)
            u_epsilon = Rnet(uxy).float().to(device)
            u=net(uxy).float().to(device)
            cond =((50*x1**4-100*x1**3-2937.5*x1**2+2987.5*x1-450)*(x1-0.5)**2+(50*y1**4-100*y1**3-2937.5*y1**2+2987.5*y1-450)*(y1-0.5)**2).float().to(device)
            loss_interior = loss_function(- gradients(u_epsilon+u, x1, 2)-gradients(u_epsilon+u, y1, 2)+alpha*(u_epsilon+u)-cond, cond1)
            return loss_interior

    def l_down1(x2,y2,cond2,Rnet,net):
            # 损失函数L2
            # x_down, y_down, cond_down = trainingdata.down(B_p)
            # data_gen=TensorDataset(x1,y1)
            # dataset = DataLoader(dataset = data_gen,
            #              batch_size = 32,
            #              shuffle=True)
            # u = net(dataset).float().to(device)
            uxy = torch.cat([x2, y2], dim=1)
            u_epsilon = Rnet(uxy).float().to(device)
            u=net(uxy).float().to(device)
            loss_down = loss_function(-gradients(u_epsilon+u,y2,1)+g*psi_t(u_epsilon+u,epsilon), cond2)
            return loss_down


    def l_up1(x3,y3,cond3,Rnet,net):
            # 损失函数L3
            # x_up, y_up, cond_up = trainingdata.up(B_p) psi_t(u,epsilon)
            uxy = torch.cat([x3, y3], dim=1)
            u_epsilon = Rnet(uxy).float().to(device)
            u=net(uxy).float().to(device)
            loss_up = loss_function(gradients(u_epsilon+u,y3,1)+g*psi_t(u_epsilon+u,epsilon), cond3)
            return loss_up
        
    def l_left1(x4,y4,cond4,Rnet,net):
            # 损失函数L4
            # x_left, y_left, cond_left = trainingdata.left(B_p)
            uxy = torch.cat([x4, y4], dim=1)
            u_epsilon = Rnet(uxy).float().to(device)
            u=net(uxy).float().to(device)
            loss_left = loss_function(-gradients(u_epsilon+u,x4,1)+g*psi_t(u_epsilon+u,epsilon), cond4)
            return loss_left
        

    def l_right1(x5,y5,cond5,Rnet,net):
            # 损失函数L5
            # x_right, y_right, cond_right = trainingdata.right(B_p)
            uxy = torch.cat([x5, y5], dim=1)
            u_epsilon = Rnet(uxy).float().to(device)
            u=net(uxy).float().to(device)
            loss_right = loss_function(gradients(u_epsilon+u,x5,1)+g*psi_t(u_epsilon+u,epsilon), cond5)
            return loss_right

    # Compute kernel


    def loss2(x1,y1,cond1,x2,y2,cond2,x3,y3,cond3,x4,y4,cond4,x5,y5,cond5,Rnet,net):
            # all the loss loss(x1,y1,cond1,x2,y2,cond2,x3,y3,cond3,x4,y4,cond4,x5,y5,cond5,x6,y6,cond6,net):
            loss_interior1= pdeloss2.l_interior1(x1,y1,cond1,Rnet,net)
            loss_down1 = pdeloss2.l_down1(x2,y2,cond2,Rnet,net)
            loss_up1 = pdeloss2.l_up1(x3,y3,cond3,Rnet,net)
            loss_left1 = pdeloss2.l_left1(x4,y4,cond4,Rnet,net)
            loss_right1 = pdeloss2.l_right1(x5,y5,cond5,Rnet,net)
            
            loss_val1 = 1*(loss_interior1)+1*(loss_down1+loss_up1+loss_left1+loss_right1)
            
            # +10**(-4)*loss_grad
            return loss_val1

    def closure2():
                
                    
        loss_2 = pdeloss2.loss2(x_interior1, y_interior1, cond_interior1,x_down1, y_down1, cond_down1,x_up1, y_up1, cond_up1, x_left1, y_left1, cond_left1,x_right1, y_right1, cond_right1,Rnet,net)
        optimizer1.zero_grad()
        
        # loss_1.backward()  loss_1 = loss(x_interior, y_interior, cond_interior,x_down, y_down, cond_down,x_up, y_up, cond_up, x_left, y_left, cond_left,x_right, y_right, cond_right,x_data_interior, y_data_interior, cond_data_interior,net)  
        loss_2.backward()
        
        return loss_1

# Compute kernel





# define self-adaptive importance sampling




####Solution Plot
""" def solutionplot(u_pred_fig,u_error_fig,u_real_fig):
    
    fig, ax = plt.subplots()
    fig.suptitle('Numerical results obtained by SAIS',fontsize = 12)
    ax.axis('off')
    
    gs = gridspec.GridSpec(2, 2)
    
    gs.update(top=1-0.07, bottom=0.1, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs[:, 0])

    h = ax.imshow((u_pred_fig).detach().cpu().numpy(), interpolation='nearest', cmap='rainbow', 
                extent=[xm_test.detach().cpu().numpy().min(), xm_test.detach().cpu().numpy().max(), ym_test.detach().cpu().numpy().min(), ym_test.detach().cpu().numpy().max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    
    # ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

    
    # ####### Row 1: u(t,x) slices ##################

    
    ax = plt.subplot(gs[0, 1])

    h = ax.imshow((u_error_fig).detach().cpu().numpy(), interpolation='nearest', cmap='rainbow', 
                extent=[xm_test.detach().cpu().numpy().min(), xm_test.detach().cpu().numpy().max(), ym_test.detach().cpu().numpy().min(), ym_test.detach().cpu().numpy().max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    
   
    ax = plt.subplot(gs[1, 1])
    ax.plot(xc_test.detach().cpu().numpy(),u_real_fig.detach().cpu().numpy().T[750,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(xc_test.detach().cpu().numpy(),u_pred_fig.detach().cpu().numpy().T[750,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(0.5,y)$')

    ax.set_xlim([0,1.1])
    ax.set_ylim([-0.5,0.1]) 
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
    plt.savefig('/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/image/3_Atest.jpg',dpi = 500)
    # 画loss曲线
    # plt.figure()
    # ax=plt.gca()
    # # ax = plt.subplot(gs[1, 1])
    # ax.plot(range(len(loss_all)),loss_all,'b-', linewidth = 2, label = 'Loss')
    # ax.plot(range(len(L2_error)),L2_error,'r--', linewidth = 2, label = 'L2 error')
    # ax.set_xlabel('$epochs$')
    # ax.set_yscale("log")
    # ax.legend(prop = {'size':6})
    # plt.savefig('/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/image/3_loss.jpg',dpi = 500)
 """

def solutionplot(u_pred_fig,u_error_fig,u_real_fig):
    
    fig, ax = plt.subplots()
    fig.suptitle('Numerical results obtained by SAIS',fontsize = 12)
    ax.axis('off')
    # gs = fig.add_gridspec(2,2)
    plt.figure()
    ax=plt.gca()
    h = ax.imshow((u_pred_fig).detach().cpu().numpy(), interpolation='nearest', cmap='rainbow', 
                extent=[xm_test.detach().cpu().numpy().min(), xm_test.detach().cpu().numpy().max(), ym_test.detach().cpu().numpy().min(), ym_test.detach().cpu().numpy().max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    plt.savefig('/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/image1/solve.png')
    
    # ####### Row 1: u(t,x) slices ##################
    plt.figure()
    ax=plt.gca()
   
    h = ax.imshow((u_error_fig).detach().cpu().numpy(), interpolation='nearest', cmap='rainbow', 
                extent=[xm_test.detach().cpu().numpy().min(), xm_test.detach().cpu().numpy().max(), ym_test.detach().cpu().numpy().min(), ym_test.detach().cpu().numpy().max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    plt.savefig('/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/image1/abs error.png')

    
    plt.figure()
    ax=plt.gca()
     #  h=1000, 500
    ax.plot(yc_test.detach().cpu().numpy(),u_real_fig.detach().cpu().numpy().T[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(yc_test.detach().cpu().numpy(),u_pred_fig.detach().cpu().numpy().T[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$y$')
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
    

    plt.savefig('/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/image1/examlpe1 RPINN SAIS x at 0.5.png') 





# Main pinn网格输入
#  给网络输入参数
n=20
layers = np.array([2,n,n,n,n,n,n,n,1])#5 hidden layers   神经元个数会影响解的误差大小
n2=50
layers2 = np.array([2,n2,n2,n2,n2,1])#5 hidden layers   神经元个数会影响解的误差大小
device="cuda:1"


#初始化 数据
h = 100    # 画图网格密度
#第一个网络训练点数量
Nc =5000  # 内点配置点数 500,1000,1500,2000
B_p = 800  # 边界点配置点数
#第一个网络训练点数量
Nc2=5000  # 内点配置点数 500,1000,1500,2000
B_p2 = 800  # 边界点配置点数

omega=1
N=128   # 积分估计点
alpha=0.5
g=6.25
epsilon=1e-04
# 训练代数
T_max=50000
T_max2=5000
#创建高斯分布数据点集
miu= torch.tensor([[0,0]])
Sigma= torch.tensor([[1,0],[0,1]])
""" data_interior=multivariate_truncated_normal(miu.cpu().numpy().ravel(),Sigma.cpu().numpy(),[xl,yl],[xr,yr],Nc) """
""" data_interior=generate_sobol(Nc) """
# training data
x_interior, y_interior, cond_interior = trainingdata.interior(Nc)
x_down, y_down, cond_down = trainingdata.down(B_p)
x_up, y_up, cond_up = trainingdata.up(B_p)
x_left, y_left, cond_left = trainingdata.left(B_p)
x_right, y_right, cond_right = trainingdata.right(B_p)

x_interior1, y_interior1, cond_interior1 = trainingdata1.interior1(Nc2)
x_down1, y_down1, cond_down1 = trainingdata1.down1(B_p2)
x_up1, y_up1, cond_up1 = trainingdata1.up1(B_p2)
x_left1, y_left1, cond_left1 = trainingdata1.left1(B_p2)
x_right1, y_right1, cond_right1 = trainingdata1.right1(B_p2)

# Testing data
xc_test = torch.linspace(xl, xr, h)
yc_test = torch.linspace(yl, yr, h)
xm_test, ym_test = torch.meshgrid(xc_test, yc_test)
xx_test = xm_test.reshape(-1, 1)   #转成一列
yy_test = ym_test.reshape(-1, 1)
xy11_test = torch.cat([xx_test, yy_test], dim=1).float().to(device)

# start test
""" u_real = (100*(xx_test*(xx_test-1)*(xx_test-0.5)**4+yy_test*(yy_test-1)*(yy_test-0.5)**4)).float().to(device) """
#u_real,u_realx,u_realy = high_approximate(xx_test.float().to(device),yy_test.float().to(device),u0,space)

# 方便计算
#savemat('highapprox.mat',{'u_real':u_real.cpu().detach().numpy().ravel(),'u_realx':u_realx.cpu().detach().numpy().ravel(),'u_realy':u_realy.cpu().detach().numpy().ravel()})
# 读取数据
data_real = scipy.io.loadmat('/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/data/highapprox.mat')  # note change
u_real=data_real['u_real'].astype(float)
u_realx=data_real['u_realx'].astype(float)
u_realy=data_real['u_realy'].astype(float)

u_real=torch.from_numpy(u_real.T).float().to(device)
u_realx=torch.from_numpy(u_realx.T).float().to(device)
u_realy=torch.from_numpy(u_realy.T).float().to(device) 

u_real_fig = u_real.reshape(h,h)
# start test

start_time = time.time()



""" optimizer = torch.optim.LBFGS(net.parameters(), lr=0.001, 
                                    max_iter = T_max, 
                                    max_eval = None, 
                                    tolerance_grad = 1e-05, 
                                    tolerance_change = 1e-09, 
                                    history_size = 100, 
                                    line_search_fn = 'strong_wolfe')
 """

# 后验误差估计数据
data=generate_sobol(N)
A=xl*torch.ones([N,1]).float().to(device)
B=xr*torch.ones([N,1]).float().to(device)
data_down=torch.cat([data[:,0][:,None],A],dim=1)
data_up=torch.cat([data[:,0][:,None],B],dim=1)
data_left=torch.cat([A,data[:,1][:,None]],dim=1)
data_right=torch.cat([B,data[:,1][:,None]],dim=1)
data_gamma=torch.cat([data_down,data_up,data_left,data_right],dim=0).to(device)
xx_Posterior=data[:,0][:,None]
yy_Posterior=data[:,1][:,None]
xx_Posterior.requires_grad_(True)
yy_Posterior.requires_grad_(True)

#u_real_Posterior,u_realx_Posterior,u_realy_Posterior = high_approximate(xx_Posterior.float().to(device),yy_Posterior.float().to(device),u0,space)
# 方便计算
#savemat('Posterior.mat',{'u_real_Posterior':u_real_Posterior.cpu().detach().numpy().ravel(),'u_realx_Posterior':u_realx_Posterior.cpu().detach().numpy().ravel(),'u_realy_Posterior':u_realy_Posterior.cpu().detach().numpy().ravel()})
# 读取数据
data_Posterior = scipy.io.loadmat('/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/data/Posterior.mat')  # note change
u_real_Posterior=data_Posterior['u_real_Posterior'].astype(float)
u_realx_Posterior=data_Posterior['u_realx_Posterior'].astype(float)
u_realy_Posterior=data_Posterior['u_realy_Posterior'].astype(float)


u_real_Posterior=torch.from_numpy(u_real_Posterior.T).float().to(device)
u_realx_Posterior=torch.from_numpy(u_realx_Posterior.T).float().to(device)
u_realy_Posterior=torch.from_numpy(u_realy_Posterior.T).float().to(device)
""" max_iter = epochs

    

for i in range(max_iter):
    optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    net.to(device)
    loss_1 = loss(x_interior, y_interior, cond_interior,x_down, y_down, cond_down,x_up, y_up, cond_up, x_left, y_left, cond_left,x_right, y_right, cond_right,x_data_interior, y_data_interior, cond_data_interior,net)

    loss_1.backward() #backprop

    optimizer.step()
    
    # if i % (max_iter/10) == 0:

        

    #     print(i)


net.to(device) """

# 初始化loss存储

error=1
Posterior_error=0
gamma_0=0.25
h1=0.0125
while error >gamma_0**2+h1**2+Posterior_error:
        #第一个网络
    loss_all=[]
    L2_error=[]
    #第二个网络
    loss_all1=[]
    L2_error1=[]
    net = PINNNet(layers).to(device)
    # 定义优化器和学习率
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001,betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    #定义学习率调度器
    """ scheduler=lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.9) """

    """ # 导入之前的模型
    model=PINNNet(layers).to(device)
    model.load_state_dict(torch.load("Graduation_Thesis/seondEVI/example1/model/PINN_modeltestor1_1.pkl")) """
    # 新的网络
    Rnet = Newnet(layers2).to(device)
    # 定义优化器和学习率
    optimizer1 = torch.optim.Adam(Rnet.parameters(), lr=0.0001,betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    # 第一个网络训练
    for i in range(T_max):
    
            loss_1=optimizer.step(pdeloss1.closure)
            loss_all.append(loss_1.cpu().detach().numpy())
            # net.to(device)
            u_pred = net(xy11_test).float().to(device)
            error_vec=torch.linalg.norm((u_real - u_pred), 2) / torch.linalg.norm(u_real,2)
            L2_error.append(error_vec.cpu().detach().numpy())
            plt.figure()
            plt.plot(range(len(loss_all)),loss_all,'b-', linewidth = 2, label = 'Loss')
            plt.plot(range(len(L2_error)),L2_error,'r--', linewidth = 2, label = 'L2 error')
            plt.xlabel('$epochs$')
            plt.yscale("log")
            plt.legend(prop = {'size':6})
            plt.savefig("/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/image1/pinnloss32.png")
            plt.close()
            u_pred_fig = u_pred.reshape(h,h)

            plt.figure()
            plt.plot(yc_test.detach().cpu().numpy(),u_real_fig.detach().cpu().numpy().T[50,:], 'b-', linewidth = 2, label = 'Exact')       
            plt.plot(yc_test.detach().cpu().numpy(),u_pred_fig.detach().cpu().numpy().T[50,:], 'r--', linewidth = 2, label = 'Prediction')
            plt.xlabel('$y$')
            plt.ylabel('$u(0.5,y)$')
            # plt.xlim(0,1.1)
            # plt.ylim(-0.5,0.1)
            plt.savefig("/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/image1/pinn44.png")
            plt.close()
    for i in range(T_max2):
   
        loss_2=optimizer1.step(pdeloss2.closure2)
       
        # loss_1=optimizer.step(closure)
        loss_all1.append(loss_2.cpu().detach().numpy())
       
        # net.to(device)
        u_pred = (net(xy11_test)+Rnet(xy11_test)).float().to(device)
        error_vec1=torch.linalg.norm((u_real - u_pred), 2) / torch.linalg.norm(u_real,2)
        L2_error1.append(error_vec1.cpu().detach().numpy())
        plt.figure()
        plt.plot(range(len(loss_all1)),loss_all1,'b-', linewidth = 2, label = 'Loss1')
        plt.plot(range(len(L2_error1)),L2_error1,'r--', linewidth = 2, label = 'L2 error')
        plt.xlabel('$epochs$')
        plt.yscale("log")
        plt.legend(prop = {'size':6})
        plt.savefig("/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/image1/newloss32.png")
        plt.close()
        u_pred_fig = u_pred.reshape(h,h)

        plt.figure()
        plt.plot(yc_test.detach().cpu().numpy(),u_real_fig.detach().cpu().numpy().T[50,:], 'b-', linewidth = 2, label = 'Exact')       
        plt.plot(yc_test.detach().cpu().numpy(),u_pred_fig.detach().cpu().numpy().T[50,:], 'r--', linewidth = 2, label = 'Prediction')
        plt.xlabel('$y$')
        plt.ylabel('$u(0.5,y)$')
        # plt.xlim(0,1.1)
        # plt.ylim(-0.5,0.1)
        plt.savefig("/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/image1/newpinn44.png")
        plt.close()
    # 计算后验误差指标
    u_pred_Posterior =(net(torch.cat([xx_Posterior,yy_Posterior],dim=1))+Rnet(torch.cat([xx_Posterior,yy_Posterior],dim=1))).float().to(device)
    u_pred_Posterior.requires_grad_(True)
    # e_gradx=gradients(u_pred,xx,1)-u_realx
    # e_grady=gradients(u_pred,yy,1)-u_realy
    """  error=(loss_function((gradients(u_pred_Posterior,xx_Posterior,1)), u_realx)+loss_function((gradients(u_pred_Posterior,yy_Posterior,1)), u_realy))+loss_function(u_pred_Posterior,u_real_Posterior) """
    error=(loss_function((gradients(u_pred_Posterior,xx_Posterior,1)), u_realx_Posterior)+loss_function((gradients(u_pred_Posterior,yy_Posterior,1)), u_realy_Posterior))+loss_function(u_pred_Posterior,u_real_Posterior)
    sum=0
    for i in range(data_gamma.shape[0]):
            xx_p=data_gamma[i,0]
            yy_p=data_gamma[i,1]
            xy_p=torch.cat([xx_p.reshape(1,1),yy_p.reshape(1,1)],dim=1).float().to(device)
            u=net(xy_p)+Rnet(xy_p)
            sum=sum+torch.abs(u)*epsilon**2/((u)**2+epsilon**2+torch.abs(u)*torch.sqrt((u)**2+epsilon**2))
    Posterior_error=g*sum/data_gamma.shape[0]
    epsilon=0.5*epsilon
    print(epsilon,Posterior_error)
    print(error,gamma_0**2+h1**2+Posterior_error)
# 数值解图像
plt.figure()
ax3 = plt.axes(projection='3d')
u_pred =net(xy11_test)+Rnet(xy11_test)
u_pred_fig = u_pred.reshape(h,h)
ax3.plot_surface(xm_test.detach().cpu().numpy(),ym_test.detach().cpu().numpy(),u_pred_fig.detach().cpu().numpy(),cmap='rainbow')
plt.savefig("/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example2/image1/RPINNszj.png")
# torch.save(net.state_dict(),"/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example3/model/PINN_modeltestor3_{}.pkl".format(3))

elapsed = time.time() - start_time                
print('Training time: %.2f' % (elapsed))
# start test
u_pred = net(xy11_test)+Rnet(xy11_test)
u_error = (u_pred-u_real).float().to(device)

# plot
u_pred_fig = u_pred.reshape(h,h)
# u_real_fig = u_real.reshape(h,h)
u_error_fig = u_error.reshape(h,h)
print("L_2 error is: ",float(torch.linalg.norm((u_real - u_pred), 2) / torch.linalg.norm(u_real,2)))

solutionplot(u_pred_fig,u_error_fig,u_real_fig)






