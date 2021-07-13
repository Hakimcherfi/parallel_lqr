import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg

dt = 0.01

x0 = np.array([[0.],[0.],[0.],[0.]]) #contrainte
xtarg = np.array([[1.],[2.],[0.],[0.]]) #fonction cout
xterm = np.array([[1.],[2.],[0.],[0.]]) #supposition

xweight = 1.
uweight = 1.

nq = 2
nv = 2
n = nq+nv
m = 2
dimspace = 2 #pour affichage

#dynamique
Fx = np.eye(n)
Fx[:nv,nv:]=dt*np.eye(nv)
Fu = np.concatenate([0.5*dt**2*np.eye(nv),dt*np.eye(nv)])
f1 = np.zeros((n,1))

def null_space(A, eps=1e-10):
    """
    Returns an orthonormal basis of the null space of A, made of vectors whose singular values
    are close to zero (smaller than eps)
    """
    u,s,v = np.linalg.svd(A)
    m = A.shape[0]
    n = A.shape[1]
    if m<n:
        s = np.concatenate([s,np.zeros((n-m,))])
    null_mask = (s <= eps)
    null_space = np.compress(null_mask, v, axis=0)
    return np.transpose(null_space)

def orth(A,eps=1e-10):
    """
    Returns an orthonormal basis of the range space of A, made of vectors whose singular values
    are strictly greater than eps
    """
    u,s,v = np.linalg.svd(A)
    notnull_mask = (s>eps)
    return np.compress(notnull_mask,u,axis=1)

def costx(x):
    Cx = xweight*np.eye(n)
    return 0.5*(x-xtarg).T@Cx@(x-xtarg)

def costu(u):
    Cu = uweight*np.eye(m)
    return 0.5*u.T@Cu@u

def cost(x,u):
    return costx(x)+costu(u)

def Qxx(x=np.zeros((n,1))):
    return hessian(costx,x)

def Quu(u=np.zeros((m,1))):
    return hessian(costu,u)

def Qux(x=np.zeros((n,1)),u=np.zeros((m,1))):
    return np.zeros((m,n))

def qx():
    return np.zeros((n,1))

def qu():
    return np.zeros((m,1))

def jacobian(f,x,eps=1.e-4):
    """
    Returns the jacobian matrix of a function f at a given point x
    usage :
    
    def myfunction(x): #x has to be an (n,1) array and myfunction has to return an (m,1) array
        return np.concatenate([x[0:1,:]**2 + x[1:2,:]**2,x[0:1,:]**2 - x[1:2,:]**2])
    
    x = np.array([[0.],[0.]])
    myjacobian = jacobian(myfunction,x)
    """    
    m = f(x).shape[0]
    n = x.shape[0]
    jacob = np.zeros((m,n))
    for l in range(n):
        h = np.zeros((n,1))
        h[l:l+1,:] = eps
        jacob[:,l:l+1]=(f(x+h)-f(x-h))/(2.*eps)
    return jacob

def hessian(f,x,eps=1.e-4):
    """
    Returns the hessian matrix of a function f at a given point x
    usage :
    
    def myfunction(x): #x has to be an (n,1) array and myfunction has to return a (1,1) array
        return x[0:1,:]**2 + x[1:2,:]**2
    
    x = np.array([[0.],[0.]])
    myhessian = hessian(myfunction,x)
    """
    dim = x.shape[0]
    hess = np.zeros((dim,dim))
    for n in range(dim): #diagonale
        h = np.zeros((dim,1))
        h[n:n+1,:]=eps
        hess[n:n+1,n:n+1] = (f(x+h)+f(x-h)-2.*f(x))/(eps**2.)
    for n in range(dim): #hors diagonale
        for m in range(n+1,dim):
            h = np.zeros((dim,1))
            h[n:n+1,:]=eps
            h[m:m+1,:]=eps
            hess[n:n+1,m:m+1]=0.5*((f(x+h)+f(x-h)-2.*f(x))/(eps**2.)-hess[n:n+1,n:n+1]-hess[m:m+1,m:m+1])
            hess[m:m+1,n:n+1]=hess[n:n+1,m:m+1]
    return hess

def subgains(T):
    """
    Returns gains as lists of arrays from t = 0 to t = T-1 
    """
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
        Py = orth(Nu.T)
        A = Py@linalg.pinv(Nu@Py)
        B = Zw@linalg.inv(Zw.T@Muu@Zw)@(Zw.T)
        
        #calcul gains
        Kx = -A@Nx -B@Mux
        Kz = -A@Nz -B@(Mzu.T)
        k1 = -A@n1 -B@mu1
        Kxl.append(Kx)
        Kzl.append(Kz)
        k1l.append(k1)
        
        #maj cout
        Vxx = Mxx+Mux.T@Kx+Kx.T@Mux+Kx.T@Muu@Kx
        Vzx = Mzx+Mzu@Kx+Kz.T@Mux+Kz.T@Muu@Kx
        Vzz = Mzz+Kz.T@Mzu.T+Mzu@Kz+Kz.T@Muu@Kz
        vx1 = mx1+Kx.T@mu1+(Mux.T+Kx.T@Muu)@k1
        vz1 = mz1+Mzu@k1+Kz.T@mu1+Kz.T@Muu@k1
        Hx = (np.eye(n)-Nu@A)@Nx
        Hz = (np.eye(n)-Nu@A)@Nz
        h1 = (np.eye(n)-Nu@A)@n1
        
    Kxl.reverse()
    Kzl.reverse()
    k1l.reverse()
    return Kxl,Kzl,k1l

def RS(Kx,Kz,k1,T):
    """
    Returns R and S matrices for the forward pass as lists of arrays
    """
    Ra = [np.eye(n)]
    Rz = [np.zeros((n,n))]
    r1 = [np.zeros((n,1))]
    Sa = []
    Sz = []
    s1 = []
    for t in range(T):
        Sa.append(Kx[t]@Ra[t])
        Sz.append(Kx[t]@Rz[t]+Kz[t])
        s1.append(Kx[t]@r1[t]+k1[t])
        Ra.append(Fx@Ra[t]+Fu@Sa[t])
        Rz.append(Fx@Rz[t]+Fu@Sz[t])
        r1.append(Fx@r1[t]+Fu@s1[t]+f1)
    return Ra,Rz,r1,Sa,Sz,s1

def AAT(T):
    """
    Returns the A@(A.T) matrix given in equation 34
    """
    mat = np.eye((T+2)*n)
    for t in range(T):
        mat[n*t:n*(t+1),n*(t+1):n*(t+2)]=-Fx.T
        mat[n*(t+1):n*(t+2),n*t:n*(t+1)]=-Fx
        mat[n*(t+1):n*(t+2),n*(t+1):n*(t+2)]=Fx@(Fx.T)+Fu@(Fu.T)+np.eye(n)
    mat[n*T:n*(T+1),n*(T+1):]=np.eye(n)
    mat[n*(T+1):,n*T:n*(T+1)]=np.eye(n)
    return mat

def Ab(Ra,Rz,r1,Sa,Sz,s1,T):
    """
    Returns the colums in equation 36 : Da,Dz and d1 (note : on the paper,
    n rows are missing (ie (T+2)*n rows instead of (T+1)*n)
    """
    Da = np.zeros(((T+2)*n,n))
    Dz = np.zeros(((T+2)*n,n))
    d1 = np.zeros(((T+2)*n,1))
    Da[:n,:] = -Qxx()@Ra[0]-(Qux().T)@Sa[0]
    Dz[:n,:] = -Qxx()@Rz[0]-(Qux().T)@Sz[0]
    d1[:n,:] = -Qxx()@r1[0]-(Qux().T)@s1[0]-qx()
    for t in range(T-1):
        Da[n*(t+1):n*(t+2),:] = Fx@(Qxx()@Ra[t]+Qux().T@Sa[t])+Fu@(Qux()@Ra[t]+Quu()@Sa[t])-Qxx()@Ra[t+1]-Qux().T@Sa[t+1]
        Dz[n*(t+1):n*(t+2),:] = Fx@(Qxx()@Rz[t]+Qux().T@Sz[t])+Fu@(Qux()@Rz[t]+Quu()@Sz[t])-Qxx()@Rz[t+1]-Qux().T@Sz[t+1]
        d1[n*(t+1):n*(t+2),:]  = Fx@(Qxx()@r1[t]+Qux().T@s1[t]+qx())+Fu@(Qux()@r1[t]+Quu()@s1[t]+qu())-Qxx()@r1[t+1]-Qux().T@s1[t+1]-qx()
    #derniere ligne de b : pas de Qux !
    Da[n*T:n*(T+1),:] = Fx@(Qxx()@Ra[T-1]+Qux().T@Sa[T-1])+Fu@(Qux()@Ra[T-1]+Quu()@Sa[T-1])-Qxx()@Ra[T]
    Dz[n*T:n*(T+1),:] = Fx@(Qxx()@Rz[T-1]+Qux().T@Sz[T-1])+Fu@(Qux()@Rz[T-1]+Quu()@Sz[T-1])-Qxx()@Rz[T]
    d1[n*T:n*(T+1),:]  = Fx@(Qxx()@r1[T-1]+Qux().T@s1[T-1]+qx())+Fu@(Qux()@r1[T-1]+Quu()@s1[T-1]+qu())-Qxx()@r1[T]-qx()
    
    Da[n*(T+1):,:] = -Qxx()@Ra[T]
    Dz[n*(T+1):,:] = -Qxx()@Rz[T]
    d1[n*(T+1):,:] = -Qxx()@r1[T]-qx()
    return Da,Dz,d1
        
def lines(x,u,T):
    t = np.linspace(0,T[-1]*dt,T[-1]+1)
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,sharex=True)
    for k in range(dimspace):
        ax1.plot(t,x[k,:],label="x"+str(k+1))
        ax2.plot(t,x[dimspace+k,:],label="v"+str(k+1))
        ax3.plot(t[:-1],u[k,:],label="u"+str(k+1))
    for k in range(len(T)):
        ax1.axvline(T[k]*dt,color="g",linestyle="--")
        ax2.axvline(T[k]*dt,color="g",linestyle="--")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig.suptitle("Courbes")
    plt.savefig("courbes.png",dpi=1000)
    
def calculparalleleback(t0,T):
    """
    Returns the coefficients used to create the system explained after equation 7,
    for one constrained subproblem
    """
    Kxl,Kzl,k1l = subgains(T) #utiliser t0 quand la dynamique varie
    Ra,Rz,r1,Sa,Sz,s1 = RS(Kxl,Kzl,k1l,T)
    aat = AAT(T) #utiliser t0 quand la dynamique varie
    Da,Dz,d1 = Ab(Ra,Rz,r1,Sa,Sz,s1,T) #utiliser t0 quand la dynamique varie
    return aat,Da,Dz,d1,Ra,Rz,r1,Sa,Sz,s1

def calculparalleleforw(t1,t2,xinit,xterm,Ra,Rz,r1,Sa,Sz,s1):
    """
    Computes the trajectory of one subproblem
    """
    x = np.zeros((n,t2-t1+1))
    u = np.zeros((m,t2-t1))
    for t in range(t2-t1):
        x[:,t:t+1]=Ra[t]@xinit+Rz[t]@xterm+r1[t]
        u[:,t:t+1]=Sa[t]@xinit+Sz[t]@xterm+s1[t]
    x[:,-1:]=Ra[-1]@xinit+Rz[-1]@xterm+r1[-1]
    return x,u

def constrainedLQR(x0,xterm,T):
    """
    Main function, computes the gains for each sub-problem (possibly in parallel),
    solves the link points, computes the trajectory (possibly in parallel) and displays
    """
    aat,Da,Dz,d1 = [],[],[],[]
    Ra,Rz,r1,Sa,Sz,s1 = [],[],[],[],[],[]
    for k in range(len(T)-1):
        aat2,Da2,Dz2,d12,Ra2,Rz2,r12,Sa2,Sz2,s12 = calculparalleleback(0,T[k+1]-T[k])
        aat.append(aat2)
        Da.append(Da2)
        Dz.append(Dz2)
        d1.append(d12)
        Ra.append(Ra2)
        Rz.append(Rz2)
        r1.append(r12)
        Sa.append(Sa2)
        Sz.append(Sz2)
        s1.append(s12)
    xlink = solveur(aat,Da,Dz,d1,x0,xterm,T) #solves for the link points, but has to be reshaped first
    xlink = np.reshape(xlink,(len(T)-2,n))
    xlink = list(xlink)
    xlink = [np.reshape(np.array([xlink[k]]),(n,1)) for k in range (len(xlink))] #done with reshaping
    for k in range(len(xlink)):
        print("xlink({}) = \n{}".format(T[k+1]*dt,xlink[k]))
    xlink.insert(0,x0)
    xlink.append(xterm)
    x = np.zeros((n,T[-1]+1))
    u = np.zeros((m,T[-1]))
    for k in range(len(T)-1):
        x[:,T[k]:T[k+1]+1],u[:,T[k]:T[k+1]] = calculparalleleforw(T[k],T[k+1],xlink[k],xlink[k+1],Ra[k],Rz[k],r1[k],Sa[k],Sz[k],s1[k]) 
    lines(x,u,T)
    
def solveur(aat,Da,Dz,d1,x0,xterm,T):
    """
    Creates the system to find the link points and solves it
    input : aat,Da,Dz,d1 : list of arrays, one element per subproblem
    T : list of node numbers
    """
    A = np.eye((len(T)-2)*n)
    b = np.zeros(((len(T)-2)*n,1))
    for k in range(len(T)-1):
        assert(T[k+1]-T[k]>=n)
    for k in range(len(T)-2):
        A[k*n:(k+1)*n,k*n:(k+1)*n] = np.linalg.inv(aat[k])[-n:,:]@Dz[k]+np.linalg.inv(aat[k+1])[:n,:]@Da[k+1]
        b[k*n:(k+1)*n,:] = np.linalg.inv(aat[k])[-n:,:]@d1[k]+np.linalg.inv(aat[k+1])[:n,:]@d1[k+1]
    for k in range(len(T)-3):
        A[k*n:(k+1)*n,(k+1)*n:(k+2)*n] = np.linalg.inv(aat[k+1])[:n,:]@Dz[k+1]
        A[(k+1)*n:(k+2)*n,k*n:(k+1)*n] = np.linalg.inv(aat[k+1])[-n:,:]@Da[k+1]
    b[:n,:]=b[:n,:]+np.linalg.inv(aat[0])[-n:,:]@Da[0]@x0
    b[-n:,:]=b[-n:,:]+np.linalg.inv(aat[-1])[:n,:]@Dz[-1]@xterm
    return np.linalg.solve(A,-b)

constrainedLQR(x0,xterm,[0,50,100])

