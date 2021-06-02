import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
from scipy.linalg import null_space,orth
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

xterm = np.array([[0.5],[0.],[1.],[0.]]) #supposition
                
def costx(x):
    Cx = xweight*np.eye(n)
    return 0.5*(x-xtarg).T@Cx@(x-xtarg)

def costu(u):
    Cu = uweight*np.eye(m)
    return 0.5*u.T@Cu@u

def cost(x,u):
    return costx(x)+costu(u)

def finalcost(x):
    CxT = xweightT*np.eye(n)
    return 0.5*(x-xtarg).T@CxT@(x-xtarg)

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


def subproblem(xinit,xterm,T):
    
    x=np.tile(xinit,(1,T+1))
    x[:,T:T+1] = xterm
    u=np.zeros((m,T))
    
    # a t = T (T du sous probleme)
    Hx = [np.eye(n)]
    h1 = [np.zeros((n,1))]
    vx1 = [np.zeros((n,1))]
    Vzx = [np.zeros((n,n))]
    Hz = [-1.*np.eye((n))]
    Vxx = [hessien(finalcost)(xterm)]
    vz1 = [np.zeros((n,1))]
    Vzz = [np.zeros((n,n))]
    
    mx1 = []
    mz1 = []
    Mzu = []
    Mxx = []
    Mux = []
    Nu = []
    Nz = []
    mu1 = []
    Mzx = []
    Mzz = []
    Muu = []
    Nx = []
    n1 = []
    Zw = []
    Py = []
    y = []
    w = []
    Kx = []
    Kz = []
    k1 = []
    
    for t in range(T-1,-1,-1):
        # a t < T
        mx1.append(Fx.T@vx1[-1]) # + qx1T
        mz1.append(vz1[-1])
        Mzu.append(np.zeros((m,n)))
        Mxx.append(Qxx(x[:,t:t+1]) + Fx.T@Vxx[-1]@Fx)
        Mux.append(Fu.T@Vxx[-1]@Fx) # + Quxt
        Nu.append(Hx[-1]@Fu)
        Nz.append(Hz[-1])
        mu1.append(Fu.T@vx1[-1]) #+qu1t
        Mzx.append(Vzx[-1])
        Mzz.append(Vzz[-1])
        Muu.append(Quu(u[:,t:t+1]) + Fu.T@Vxx[-1]@Fu)
        Nx.append(Hx[-1]@Fx)
        n1.append(Hx[-1]@np.zeros((n,1)) + h1[-1])
        print("t = {}".format(t))
        print(Nu[-1])
        Zw.append(null_space(Nu[-1]))
        Py.append(orth(Nu[-1].T)) #regarder numpy qr...
        if (Py[-1].size>0):
            y.append(-linalg.pinv(Nu[-1]@Py[-1])@(Nx[-1]@x[:,t:t+1]+Nz[-1]@xterm+n1[-1]))
        else:
            y.append([])
        if (Zw[-1].size>0):
            w.append(-linalg.inv(Zw[-1].T@Muu[-1]@Zw[-1])@(Zw[-1].T)@(Mux[-1]@x[:,t:t+1]+Mzu[-1].T@xterm+mu1[-1]))
        else:
            w.append([])
        Kx.append(-(Py[-1]@linalg.pinv(Nu[-1]@Py[-1])@Nx[-1]+Zw[-1]@linalg.inv(Zw[-1].T@Muu[-1]@Zw[-1])@(Zw[-1].T)@Mux[-1]))
        Kz.append(-(Py[-1]@linalg.pinv(Nu[-1]@Py[-1])@Nz[-1]+Zw[-1]@linalg.inv(Zw[-1].T@Muu[-1]@Zw[-1])@(Zw[-1].T)@Mzu[-1]))
        k1.append(-(Py[-1]@linalg.pinv(Nu[-1]@Py[-1])@n1[-1]+Zw[-1]@linalg.inv(Zw[-1].T@Muu[-1]@Zw[-1])@(Zw[-1].T)@mu1[-1]))
        u[:,t:t+1] = Kx[-1]@x[:,t:t+1]+Kz[-1]@xterm+k1[-1]
        print(Kx[-1])
        print(u)
        Hx.append((np.eye(n)-Nu[-1]@Py[-1]@linalg.pinv(Nu[-1]@Py[-1]))@Nx[-1])
        Hz.append((np.eye(n)-Nu[-1]@Py[-1]@linalg.pinv(Nu[-1]@Py[-1]))@Nz[-1])
        h1.append((np.eye(n)-Nu[-1]@Py[-1]@linalg.pinv(Nu[-1]@Py[-1]))@n1[-1])
        Vxx.append(Mxx[-1]+2.*Mux[-1].T@Kx[-1]+Kx[-1].T@Muu[-1]@Kx[-1])
        Vzz.append(Mzz[-1]+2.*Mux[-1].T@Kz[-1]+Kz[-1].T@Muu[-1]@Kz[-1])
        Vzx.append(Mzx[-1]+Mzu[-1].T@Kx[-1]+Kz[-1].T@Mux[-1] +Kz[-1].T@Muu[-1]@Kx[-1])
        vx1.append(mx1[-1]+Kx[-1].T@mu1[-1]+(Mux[-1].T+Kx[-1].T@Muu[-1])@k1[-1])
        vz1.append(mz1[-1]+Mzu[-1].T@k1[-1]+Kz[-1].T@mu1[-1]+Kz[-1].T@Muu[-1]@k1[-1])
    
subproblem(x0,xtarg,T)
