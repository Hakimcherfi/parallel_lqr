import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
%matplotlib inline

T = 100 #T+1 points
dt = 0.01
x0 = np.array([[0.],[0.],[0.],[0.]]) #contrainte
xtarg = np.array([[1.],[0.],[0.],[0.]])
xweight = 0.1
uweight = 1.
xweightT = 1000.
J = 2
tau = int(T/2)

nq = 2
nv = 2
n = nq+nv
m = 2
Fx = np.eye(n)
Fx[:nv,nv:]=dt*np.eye(nv)
Fu = np.concatenate([0.5*dt**2*np.eye(nv),dt*np.eye(nv)])

xlink = np.array([[0.5],[0.],[1.],[0.]]) #supposition


                
def costx(x):
    Cx = xweight*np.eye(n)
    return 0.5*(x-xtarg).T@Cx@(x-xtarg)

def costu(u):
    Cu = uweight*np.eye(m)
    return 0.5*u.T@Cu@u

def cost(x,u):
    return costx(x)+costu(u)

def next_state(x,u):
    return Fx@x + Fu@u
    
def Qxx(x=np.zeros((n,1))):
    return hessien(costx)(x)

def Quu(u):
    return hessien(costu)(u)

def gradient(f):
    def fbis(x,eps=0.0001):
        dim = x.shape[0]
        grad = np.zeros((dim,1))
        for n in range(dim):
            h = np.zeros((dim,1))
            h[n:n+1,:] = eps
            grad[n:n+1,:]=((f(x+h)-f(x-h))/(2*eps))
        return grad
    return fbis

def hessien(f):
    def fbis(x,eps=0.0001):
        dim = x.shape[0]
        hess = np.zeros((dim,dim))
        for n in range(dim):
            h = np.zeros((dim,1))
            h[n:n+1,:]=eps
            hess[n:n+1,n:n+1] = (f(x+h)+f(x-h)-2*f(x))/(eps**2)
        for n in range(dim):
            for m in range(n+1,dim):
                h = np.zeros((dim,1))
                h[n:n+1,:]=eps
                h[m:m+1,:]=eps
                hess[n:n+1,m:m+1]=0.5*((f(x+h)+f(x-h)-2*f(x))/(eps**2)-hess[n:n+1,n:n+1]-hess[m:m+1,m:m+1])
                hess[m:m+1,n:n+1]=hess[n:n+1,m:m+1]
        return hess
    return fbis


def subproblem():
    # a t = T (T du sous probleme)
    HxT = np.eye(n)
    h1T = np.zeros((n,1))
    vx1T = np.zeros((n,1))
    VzxT = np.zeros((n,n))
    HzT = -1.*np.eye((n))
    VxxT = Qxx()
    vz1T = np.zeros((n,1))
    VzzT = np.zeros((n,n))
    
    # a t = T-1  
    mx1t = Fx.T@vx1T # + qx1T
    mz1t = vz1T
    Mzut = np.zeros((n,m))
    Mxxt = Qxx() + Fx.T@VxxT@Fx
    Muxt = Fu.T@VxxT@Fx # + Quxt
    Nut = HxT@Fu
    Nzt = Hzt
    mu1t = Fu.T@vx1T #+qu1t
    Mzxt = Vzxt
    Mzzt = Vzzt
    Muut = Quu(ut) + Fu.T@VxxT@Fu
    Nxt = HxT@Fx
    n1t = HxT@np.zeros((n,1)) + h1T

subproblem()
