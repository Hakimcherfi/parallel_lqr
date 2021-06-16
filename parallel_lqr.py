import numpy as np
import pdb
from matplotlib import pyplot as plt
from numpy import linalg
import scipy.linalg
%matplotlib inline

T = 100 #T+1 points
dt = 0.01

x0 = np.array([[0.],[0.]]) #contrainte
xtarg = np.array([[1.],[0.]]) #fonction cout
xterm = np.array([[1.],[0.]]) #supposition

xweight = 0.
uweight = 1.

nq = 1
nv = 1
n = nq+nv
m = 1
dimspace = 1 #pour affichage

#dynamique
Fx = np.eye(n)
Fx[:nv,nv:]=dt*np.eye(nv)
Fu = np.concatenate([0.5*dt**2*np.eye(nv),dt*np.eye(nv)])

def null_space(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    m = A.shape[0]
    n = A.shape[1]
    if m<n:
        s = np.concatenate([s,np.zeros((n-m,))])
    null_mask = (s <= eps)
    null_space = np.compress(null_mask, vh, axis=0)
    return np.transpose(null_space)

def orth(A,eps=1e-15):
    u,s,vh = scipy.linalg.svd(A)
    m = A.shape[0]
    n = A.shape[1]
    if m<n:
        s = np.concatenate([s,np.zeros((n-m,))])
    notnull_mask = (s>eps)
    orth_space = np.compress(notnull_mask,vh,axis=0)
    return np.transpose(orth_space)

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

def Quu(u=np.zeros((m,1))):
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


def subgains(xterm,T):
    # a t = T (T du sous probleme)
    Vxx = Qxx()
    Vzx = np.zeros((n,n))
    Vzz = np.zeros((n,n))
    vx1 = np.zeros((n,1)) #+qx1T
    vz1 = np.zeros((n,1))
    
    Hx = np.eye(n)
    Hz = -1.*np.eye((n))
    h1 = np.zeros((n,1))

    Kxl = []
    Kzl= []
    k1l = [] 
    
    for t in range(T-1,-1,-1):
        # a t < T
        Mxx = Qxx()+Fx.T@Vxx@Fx
        Mux = Fu.T@Vxx@Fx # + Qux
        Mzu = Vzx@Fu
        Mzx = Vzx@Fx
        Mzz = Vzz
        Muu = Quu() + Fu.T@Vxx@Fu
        mx1 = Fx.T@vx1 # + qx1
        mu1 = Fu.T@vx1 #+qu1
        mz1 = vz1
        Nx = Hx@Fx
        Nu = Hx@Fu
        Nz = Hz
        n1 = h1 #+Hx@np.zeros((n,1))
        Zw = null_space(Nu)
        Py = orth(Nu)
        A = Py@linalg.pinv(Nu@Py)
        B = Zw@linalg.inv(Zw.T@Muu@Zw)@(Zw.T)
        
        #calcul gains
        Kx = -A@Nx -B@Mux
        Kz = -A@Nz -B@(Mzu.T)
        k1 = -A@n1 -B@mu1
        Kxl.append(Kx)
        Kzl.append(Kz)
        k1l.append(k1)
        #assert Py.shape!=((m,1))
        if (False):
            print("t={}".format(t))
            print("Nu={}".format(Nu))
            print("Py={}".format(Py))
            print("Zw={}".format(Zw))
            print("A={}".format(A))
            print("B={}".format(B))
            if (Py.shape==((m,m))):
                print("Py==Im ? : {}".format((Py==np.eye(m)).all()))
            print("A*Nu == Im ? : {}".format((A@Nu==np.eye(m)).all()))
            print("A*Nu is close to Im ? : {}".format(np.allclose(A@Nu,np.eye(m))))
            print("Hxt+1 :{}".format(Hx))    
        #maj cout
        Vxx = Mxx+Mux.T@Kx+Kx.T@Mux+Kx.T@Muu@Kx
        Vzx = Mzx+Mzu@Kx+Kz.T@Mux+Kz.T@Muu@Kx
        Vzz = Mzz+Kz.T@Mzu.T+Mzu@Kz+Kz.T@Muu@Kz
        vx1 = mx1+Kx.T@mu1+(Mux.T+Kx.T@Muu)@k1
        vz1 = mz1+Mzu@k1+Kz.T@mu1+Kz.T@Muu@k1
        Hx = (np.eye(n)-Nu@A)@Nx
        Hz = (np.eye(n)-Nu@A)@Nz
        h1 = (np.eye(n)-Nu@A)@n1
    return Kxl,Kzl,k1l

def subroll(x0,xterm,Kx,Kz,k1):
    T = len(Kx)
    x = np.zeros((n,T+1))
    x[:,0:1] = x0
    x[:,-2:-1] = xterm
    u = np.zeros((m,T))
    for t in range(T):
        u[:,t:t+1] = Kx[-1]@x[:,t:t+1]+Kz[-1]@xterm+k1[-1]
        x[:,t+1:t+2] = next_state(x[:,t:t+1],u[:,t:t+1])
        Kx.pop()
        Kz.pop()
        k1.pop()
    return x,u

def constrainedLQR(x0,xterm,T):
    Kxl,Kzl,k1l = subgains(xterm,T)
    return subroll(x0,xterm,Kxl,Kzl,k1l)

def scatter_x(x,cercle=False):
    plt.figure()
    figure, axes = plt.subplots()
    if (dimspace==2):
        if(cercle):
            draw_circle = plt.Circle((xsphere[0:1,:], xsphere[1:2,:]), Rsphere,fill=False)
            draw_circle2 = plt.Circle((xsphere[0:1,:], xsphere[1:2,:]), Rsphere+distsecu,fill=False)
            axes.add_artist(draw_circle)
            axes.add_artist(draw_circle2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.scatter(x[0:1,:],x[1:2,:],s=3)
        plt.title("Trajectoire")
        plt.savefig("trajectoire.png")
    if(dimspace==1):
        plt.scatter(x[0:1,:],np.zeros((1,x[0:1,:].shape[1])))
        plt.title("Trajectoire")
        
def lines(x,u):
    t = np.linspace(0,T*dt,T+1)
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,sharex=True)
    for k in range(dimspace):
        ax1.plot(t,x[k,:],label="x"+str(k+1))
        ax2.plot(t,x[dimspace+k,:],label="v"+str(k+1))
        ax3.plot(t[:-1],u[k,:],label="u"+str(k+1))
    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig.suptitle("Courbes")
    plt.savefig("courbes.png",dpi=1000)

def calcul_cout_et_affichage(x,u):
    scatter_x(x,False)
    lines(x,u)

x,u = constrainedLQR(x0,xterm,T)
calcul_cout_et_affichage(x,u)
