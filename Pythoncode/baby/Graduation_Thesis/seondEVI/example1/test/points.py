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


xl = 0  # x in [-1,1]
xr = 1
yl = 0  # y in [-1,1]
yr = 1

# Data Prep
data_finte = scipy.io.loadmat('/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example1/data/example1.mat')  # note change
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
h = 100    # 画图网格密度
N=128
# Testing data
xc_test = torch.linspace(xl, xr, h)
yc_test = torch.linspace(yl, yr, h)
xm_test, ym_test = torch.meshgrid(xc_test, yc_test)
xx_test = xm_test.reshape(-1, 1)   #转成一列
yy_test = ym_test.reshape(-1, 1)
xy11_test = torch.cat([xx_test, yy_test], dim=1).float().to(device)

u_real,u_realx,u_realy = high_approximate(xx_test.float().to(device),yy_test.float().to(device),u0,space)

# 方便计算
savemat('/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example1/data/highapprox.mat',{'u_real':u_real.cpu().detach().numpy().ravel(),'u_realx':u_realx.cpu().detach().numpy().ravel(),'u_realy':u_realy.cpu().detach().numpy().ravel()})
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
u_real_Posterior,u_realx_Posterior,u_realy_Posterior = high_approximate(xx_Posterior.float().to(device),yy_Posterior.float().to(device),u0,space)
# 方便计算
savemat('/home/pengyaoguang/baby/Graduation_Thesis/seondEVI/example1/data/Posterior.mat',{'u_real_Posterior':u_real_Posterior.cpu().detach().numpy().ravel(),'u_realx_Posterior':u_realx_Posterior.cpu().detach().numpy().ravel(),'u_realy_Posterior':u_realy_Posterior.cpu().detach().numpy().ravel()})
print(True)