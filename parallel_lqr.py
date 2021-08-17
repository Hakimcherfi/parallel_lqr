#!/usr/local/bin/python3.8

import numpy as np
from matplotlib import pyplot as plt

#variables to define the problem
T = [0,25,50,75,100] #list of nodes (here, 4 subproblems of 25 nodes each
dt = 0.01
n = 2 #state dimension
m = 1 #control dimension
xweight = np.array([1.,1.]) #weights used by the cost function
uweight = np.array([1.]) #weights used by the cost function
xweightT = np.array([1.,1.]) #weights used by the final cost function
x0 = np.array([[1.],[0.]]) #initial state
xtarg = np.array([[0.],[0.]]) #target, used by the cost function
x = np.tile(x0,(1,T[-1]+1)) #initial guess state trajectory
u = np.zeros((m,T[-1])) #initial guess control trajectory

#linear algebra functions

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
    for n1 in range(dim):
        h1 = np.zeros((dim,1))
        h1[n1:n1+1,:]=eps
        for n2 in range(dim):
            h2 = np.zeros((dim,1))
            h2[n2:n2+1,:]=eps
            hess[n1:n1+1,n2:n2+1]=(f(x+h1+h2)-f(x+h1)-f(x+h2)+f(x))/(eps**2.)
    return hess

def null_space(A, eps=1.e-10):
    """
    Returns an orthonormal basis of the null space of A, made of vectors whose singular values
    are close to zero (smaller or equal to eps)
    """
    u,s,v = np.linalg.svd(A)
    m = A.shape[0]
    n = A.shape[1]
    if m<n:
        s = np.concatenate([s,np.zeros((n-m,))])
    null_mask = (s <= eps)
    null_space = np.compress(null_mask, v, axis=0)
    return np.transpose(null_space)

def orth(A,eps=1.e-10):
    """
    Returns an orthonormal basis of the range space of A, made of vectors whose singular values
    are strictly greater than eps
    """
    u,s,v = np.linalg.svd(A)
    notnull_mask = (s>eps)
    return np.compress(notnull_mask,u,axis=1)


#functions to return the coefficients of 1 backward pass of 1 subproblem

def calculparalleleback(x,u,t0,t1):
    """
    Returns the coefficients used to create the system explained after equation 7,
    for one constrained subproblem
    parameters : x and u (of the subproblem only),first and last nodes numbers
    """
    Kxl,Kzl,k1l = subgains(x,u,t0,t1) 
    Ra,Rz,r1,Sa,Sz,s1 = RS(x,u,Kxl,Kzl,k1l,t0,t1)
    aat = AAT(x,u,t0,t1) 
    Da,Dz,d1 = Ab(Ra,Rz,r1,Sa,Sz,s1,t0,t1,x,u) 
    return aat,Da,Dz,d1,Ra,Rz,r1,Sa,Sz,s1

def subgains(x,u,t0,t1):

    """computes the backward pass of a subproblem
    input : initial guesses x and u, first node number and last node number
    output : 3 lists of gains, list going from t0 to t1
    """

    # at t = T (ie t1)
    Vxx = Qxx(x[:,-1:],u[:,-1:])
    Vzx = np.zeros((n,n))
    Vzz = np.zeros((n,n))
    vx1 = qx(x[:,-1:],u[:,-1:])
    vz1 = np.zeros((n,1))

    Hx = np.eye(n)
    Hz = -1.*np.eye((n))
    h1 = np.zeros((n,1))

    Kxl = []
    Kzl= []
    k1l = []

    # a t < T
    for t in range(t1-t0-1,-1,-1):
        Fxt = Fx(x[:,t:t+1],u[:,t:t+1])
        Fut = Fu(x[:,t:t+1],u[:,t:t+1])   
        f1t = f1(x[:,t+1:t+2],x[:,t:t+1],u[:,t:t+1])
        Qxxt = Qxx(x[:,t:t+1],u[:,t:t+1])
        Quxt = Qux(x[:,t:t+1],u[:,t:t+1])
        qxt = qx(x[:,t:t+1],u[:,t:t+1])
        qut = qu(x[:,t:t+1],u[:,t:t+1])
        Mxx = Qxxt+Fxt.T@Vxx@Fxt
        Mux = Fut.T@Vxx@Fxt + Quxt
        Mzu = Vzx@Fut
        Mzx = Vzx@Fxt
        Mzz = Vzz
        Muu = Quu(x[:,t:t+1],u[:,t:t+1]) + Fut.T@Vxx@Fut
        Mfu = Vxx@Fut
        mx1 = Fxt.T@vx1 + qxt
        mu1 = Fut.T@vx1 + qut
        mz1 = vz1
        Nx = Hx@Fxt
        Nu = Hx@Fut
        Nz = Hz
        n1 = h1 + Hx@f1t
        Zw = null_space(Nu)
        Py = orth(Nu.T)
        A = Py@np.linalg.pinv(Nu@Py)
        B = Zw@np.linalg.inv(Zw.T@Muu@Zw)@(Zw.T)

        #calcul gains
        Kx = -A@Nx -B@Mux
        Kz = -A@Nz -B@(Mzu.T)
        k1 = -A@n1 -B@(mu1+Mfu.T@f1t)
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

def RS(x,u,Kx,Kz,k1,t0,t1):
    """
    Returns R and S matrices for the forward pass as lists of arrays
    """
    Ra = [np.eye(n)]
    Rz = [np.zeros((n,n))]
    r1 = [np.zeros((n,1))]
    Sa = []
    Sz = []
    s1 = []
    for t in range(t1-t0):
        Sa.append(Kx[t]@Ra[t])
        Sz.append(Kx[t]@Rz[t]+Kz[t])
        s1.append(Kx[t]@r1[t]+k1[t])
        Ra.append(Fx(x[:,t:t+1],u[:,t:t+1])@Ra[t]+Fu(x[:,t:t+1],u[:,t:t+1])@Sa[t])
        Rz.append(Fx(x[:,t:t+1],u[:,t:t+1])@Rz[t]+Fu(x[:,t:t+1],u[:,t:t+1])@Sz[t])
        r1.append(Fx(x[:,t:t+1],u[:,t:t+1])@r1[t]+Fu(x[:,t:t+1],u[:,t:t+1])@s1[t]+f1(x[:,t+1:t+2],x[:,t:t+1],u[:,t:t+1]))
    return Ra,Rz,r1,Sa,Sz,s1

def AAT(x,u,t0,t1):
    """
    Returns the A@(A.T) matrix given in equation 34 for a constrained LQR
    """
    T = t1-t0
    mat = np.eye((T+2)*n)
    for t in range(T):
        Fxt = Fx(x[:,t:t+1],u[:,t:t+1])
        Fut = Fu(x[:,t:t+1],u[:,t:t+1])
        mat[n*t:n*(t+1),n*(t+1):n*(t+2)]=-Fxt.T
        mat[n*(t+1):n*(t+2),n*t:n*(t+1)]=-Fxt
        mat[n*(t+1):n*(t+2),n*(t+1):n*(t+2)]=Fxt@(Fxt.T)+Fut@(Fut.T)+np.eye(n)
    mat[n*T:n*(T+1),n*(T+1):]=np.eye(n)
    mat[n*(T+1):,n*T:n*(T+1)]=np.eye(n)
    return mat

def Ab(Ra,Rz,r1,Sa,Sz,s1,t0,t1,x,u):
    """
    Returns the colums in equation 36 : Da,Dz and d1 (note : on the paper,
    n rows are missing (ie (T+2)*n rows instead of (T+1)*n)
    """
    T = t1-t0
    Da = np.zeros(((T+2)*n,n))
    Dz = np.zeros(((T+2)*n,n))
    d1 = np.zeros(((T+2)*n,1))
    #1 a n lignes
    Da[:n,:] = -Qxx(x[:,:1],u[:,:1])@Ra[0]-(Qux(x[:,:1],u[:,:1]).T)@Sa[0]
    Dz[:n,:] = -Qxx(x[:,:1],u[:,:1])@Rz[0]-(Qux(x[:,:1],u[:,:1]).T)@Sz[0]
    d1[:n,:] = -Qxx(x[:,:1],u[:,:1])@r1[0]-(Qux(x[:,:1],u[:,:1]).T)@s1[0]-qx(x[:,:1],u[:,:1])

    #n+1 a n*T lignes
    for t in range(T-1):
        Fxt = Fx(x[:,t:t+1],u[:,t:t+1])
        Fut = Fu(x[:,t:t+1],u[:,t:t+1])
        Qxxt = Qxx(x[:,t:t+1],u[:,t:t+1])
        Quxt = Qux(x[:,t:t+1],u[:,t:t+1])
        Quut = Quu(x[:,t:t+1],u[:,t:t+1])
        Qxxt1 = Qxx(x[:,t+1:t+2],u[:,t+1:t+2])
        Quxt1 = Qux(x[:,t+1:t+2],u[:,t+1:t+2])
        Da[n*(t+1):n*(t+2),:] = Fxt@(Qxxt@Ra[t]+Quxt.T@Sa[t])+Fut@(Quxt@Ra[t]+Quut@Sa[t])-Qxxt1@Ra[t+1]-Quxt1.T@Sa[t+1]
        Dz[n*(t+1):n*(t+2),:] = Fxt@(Qxxt@Rz[t]+Quxt.T@Sz[t])+Fut@(Quxt@Rz[t]+Quut@Sz[t])-Qxxt1@Rz[t+1]-Quxt1.T@Sz[t+1]
        d1[n*(t+1):n*(t+2),:] = Fxt@(Qxxt@r1[t]+Quxt.T@s1[t]+qx(x[:,t:t+1],u[:,t:t+1]))+Fut@(Quxt@r1[t]+Quut@s1[t]+qu(x[:,t:t+1],u[:,t:t+1]))-Qxxt1@r1[t+1]-Quxt1.T@s1[t+1]-qx(x[:,t+1:t+2],u[:,t+1:t+2])

    #(n*T)+1 a n*(T+1) lignes de Ab (pas de Qux !!)
    t = T-1
    Fxt = Fx(x[:,t:t+1],u[:,t:t+1])
    Fut = Fu(x[:,t:t+1],u[:,t:t+1])
    Qxxt = Qxx(x[:,t:t+1],u[:,t:t+1])
    Quxt = Qux(x[:,t:t+1],u[:,t:t+1])
    Quut = Quu(x[:,t:t+1],u[:,t:t+1])
    Qxxt1 = Qxx(x[:,t+1:t+2])
    Da[n*T:n*(T+1),:] = Fxt@(Qxxt@Ra[T-1]+Quxt.T@Sa[T-1])+Fut@(Quxt@Ra[T-1]+Quut@Sa[T-1])-Qxxt1@Ra[T]
    Dz[n*T:n*(T+1),:] = Fxt@(Qxxt@Rz[T-1]+Quxt.T@Sz[T-1])+Fut@(Quxt@Rz[T-1]+Quut@Sz[T-1])-Qxxt1@Rz[T]
    d1[n*T:n*(T+1),:] = Fxt@(Qxxt@r1[T-1]+Quxt.T@s1[T-1]+qx(x[:,t:t+1],u[:,t:t+1]))+Fut@(Quxt@r1[T-1]+Quut@s1[T-1]+qu(x[:,t:t+1],u[:,t:t+1]))-Qxxt1@r1[T]-qx(x[:,t+1:t+2])

    #n*(T+1)+1 a n*(T+2) lignes de Ab
    t = T
    Da[n*(T+1):,:] = -Qxx(x[:,t:])@Ra[T]
    Dz[n*(T+1):,:] = -Qxx(x[:,t:])@Rz[T]
    d1[n*(T+1):,:] = -Qxx(x[:,t:])@r1[T]-qx(x[:,t:])
    return Da,Dz,d1

#Functions regarding cost and dynamics for any system

def Fx(x,u):
    """
    For unicycle :
    1 0 -v*sin(theta)*dt
    0 1  v*cos(theta)*dt
    0 0  1
    """
    return jacobian(next_state_warp,np.concatenate([x,u]))[:,:n]

def Fu(x,u):
    """
    For unicycle :
    cos(theta)*dt 0
    sin(theta)*dt 0
    0             dt
    """
    return jacobian(next_state_warp,np.concatenate([x,u]))[:,n:]

def f1(x1,x,u):
    #return np.zeros((n,1))
    return x1-next_state(x,u)

def next_state(x,u):
    return next_state_warp(np.concatenate([x,u]))

def cost(x,u):
    return cost_warp(np.concatenate([x,u]))

def QxxT(x = np.zeros((n,1))):
    return hessian(costxT,x)

def qxT(x=np.zeros((n,1))):
    #return np.zeros((n,1))
    return jacobian(costxT,x).T - QxxT(x)@x

def qx(x,u=np.zeros((m,1))):
    #return np.zeros((n,1))
    a = Qxx(x,u)@x+Qux(x,u).T@u
    return jacobian(cost_warp,np.concatenate([x,u]))[:,:n].T - a

def qu(x,u=np.zeros((m,1))):
    #return np.zeros((m,1))
    a = Qux(x,u)@x+Quu(x,u)@u
    return jacobian(cost_warp,np.concatenate([x,u]))[:,n:].T - a

def Qxx(x,u=np.zeros((m,1))):
    #return xweight*np.eye(n)
    return hessian(cost_warp,np.concatenate([x,u]))[:n,:n]

def Quu(x,u):
    #return uweight*np.eye(m)
    return hessian(cost_warp,np.concatenate([x,u]))[n:,n:]

def Qux(x,u):
    #return np.zeros((m,n))
    return hessian(cost_warp,np.concatenate([x,u]))[n:,:n]


#Functions to apply the forward pass

def calculparalleleforw(t0,t1,xinit,xterm,Ra,Rz,r1,Sa,Sz,s1):
    """
    Computes the trajectory of one subproblem
    """
    x = np.zeros((n,t1-t0+1))
    u = np.zeros((m,t1-t0))
    for t in range(t1-t0):
        x[:,t:t+1]=Ra[t]@xinit+Rz[t]@xterm+r1[t]
        u[:,t:t+1]=Sa[t]@xinit+Sz[t]@xterm+s1[t]
    x[:,-1:]=Ra[-1]@xinit+Rz[-1]@xterm+r1[-1]
    return x,u

#Functions to solve the link points

def solveur(aat,Da,Dz,d1,x0,T):
    """
    Creates the system to find the link points and solves it
    input : aat,Da,Dz,d1 : list of arrays, one element per subproblem
    T : list of node numbers
    """
    A = np.eye(n*(len(T)-1))
    B = np.zeros((n*(len(T)-1),1))
    
    La0 = []
    Lz0 = []
    l10 = []
    LaT = []
    LzT = []
    l1T = []
    Ea = []
    Ez = []
    e1 = []
    
    for k in range(len(T)-1):
        La01,Lz01,l101,LaT1,LzT1,l1T1,Ea1,Ez1,e11=LandE(aat[k],Da[k],Dz[k],d1[k])
        La0.append(La01)
        Lz0.append(Lz01)
        l10.append(l101)
        LaT.append(LaT1)
        LzT.append(LzT1)
        l1T.append(l1T1)
        Ea.append(Ea1)
        Ez.append(Ez1)
        e1.append(e11)
        
    for k in range(len(T)-2): 
        #termes sur diagonale de A sauf le dernier :
        A[k*n:(k+1)*n,k*n:(k+1)*n] = Ez[k]+La0[k+1]#ok
        #termes a droite de diagonale de A :
        A[k*n:(k+1)*n,(k+1)*n:(k+2)*n] = Lz0[k+1]#ok
        #termes e sur B :
        B[k*n:(k+1)*n,:] += e1[k]#ok
        #termes l sur B
        B[k*n:(k+1)*n,:] += l10[k+1]#ok
    for k in range (len(T)-3):
        #termes a gauche de diagonale
        A[(k+1)*n:(k+2)*n,k*n:(k+1)*n] = Ea[k+1]#ok
    A[-n:,-2*n:-n] = LaT[-1]#ok
    A[-n:,-n:] = LzT[-1]+QxxT() #ok
    B[:n,:] += Ea[0]@x0 #ok
    B[-n:,:] = l1T[-1]+qxT() #ok
    xlink = np.linalg.solve(A,-B)
    return xlink

def LaLzl1(aat,Da,Dz,d1):
    """
    Rewrites the lagrange multipliers systems
    Returns La,Lz,l1
    """
    aat1 = np.linalg.inv(aat)
    return aat1@Da,aat1@Dz,aat1@d1

def LandE(aat,Da,Dz,d1):
    """
    Returns the terms used to calculate the link points :
    La0,Lz0,l10,LaT,LzT,l1T,Ea,Ez,e1
    """
    La,Lz,l1 = LaLzl1(aat,Da,Dz,d1)
    return La[:n,:],Lz[:n,:],l1[:n,:],La[-2*n:-n,:],Lz[-2*n:-n,:],l1[-2*n:-n,:],La[-n:,:],Lz[-n:,:],l1[-n:,:]

#def next_state_warp(x):
#    """
#    Dynamique unicycle
#    x = (x,y,theta) en m,m,radians
#    u = (vitesse lin,vitesse ang) en m/s,rad/s
#    """
#    #x[2:3,:]%=(2.*np.pi)
#    x1 = x[0:1,:]+x[3:4,:]*dt*np.cos(x[2:3,:])
#    y1 = x[1:2,:]+x[3:4,:]*dt*np.sin(x[2:3,:])
#    theta1 = x[2:3,:]+x[4:5,:]*dt
#    theta1%=(2.*np.pi)
#    return np.concatenate([x1,y1,theta1])


#Functions for the particular system being tested

def next_state_warp(x):
    """Dynamique pointmass zoh
    X = (x,v)
    u = acceleration
    x(k+1) = 0.5*(dt**2)*u + v(k)*dt + x(k)
    v(k+1) = v(k) + u(k)*dt
    """
    A = np.eye(n)
    A[:int(n/2),int(n/2):] = np.eye(int(n/2))*dt
    #print("A = {}".format(A))
    B = np.zeros((n,m))
    B[:int(n/2),:] = np.eye(int(n/2))*(dt**2)/2
    B[int(n/2):,:] = np.eye(int(n/2))*dt
    #print("B = {}".format(B))
    return A@x[:n,:] + B@x[n:,:]

def cost_warp(x):
    """
    Unicycle cost function
    x = (x,y,theta,lin. speed, rotat. speed)
    """
    #Cx = xweight*np.eye(n)
    #Cx[2:,2:]=0.
    #Cu = uweight*np.eye(m)
    return 0.5*(x[:n,:]-xtarg).T@np.diag(xweight)@(x[:n,:]-xtarg) + 0.5*x[n:,:].T@np.diag(uweight)@x[n:,:]

def costxT(x):
    return 0.5*(x-xtarg).T@np.diag(xweightT)@(x-xtarg)

def constrainedLQR(x,u,T):
    """
    Main function, computes the gains for each sub-problem (possibly in parallel),
    solves the link points, computes the trajectory (possibly in parallel) and displays
    """
    aat,Da,Dz,d1 = [],[],[],[]
    Ra,Rz,r1,Sa,Sz,s1 = [],[],[],[],[],[]
    for k in range(len(T)-1):
        """
        Du premier au dernier sous probleme (0 a T)
        """
        aat2,Da2,Dz2,d12,Ra2,Rz2,r12,Sa2,Sz2,s12 = calculparalleleback(x[:,T[k]:T[k+1]+1],u[:,T[k]:T[k+1]],T[k],T[k+1])
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

    xlink = solveur(aat,Da,Dz,d1,x0,T) #solves for the link points, but has to be reshaped first
    xlink = np.reshape(xlink,(len(T)-1,n))
    xlink = list(xlink)
    xlink = [np.reshape(np.array([xlink[k]]),(n,1)) for k in range (len(xlink))] #done with reshaping
    for k in range(len(xlink)):
        print("xlink({}) = \n{}".format(T[k+1]*dt,xlink[k]))
    xlink.insert(0,x0)
    x = np.zeros((n,T[-1]+1))
    u = np.zeros((m,T[-1]))
    for k in range(len(T)-1):
        x[:,T[k]:T[k+1]+1],u[:,T[k]:T[k+1]] = calculparalleleforw(T[k],T[k+1],xlink[k],xlink[k+1],Ra[k],Rz[k],r1[k],Sa[k],Sz[k],s1[k]) 
    #lines(x,u,T)
    return x,u

#Display functions

#def plotUnicycle(x):
#    sc, delta = .1, .1
#    a, b, th = np.asscalar(x[0:1,:].reshape(1,)), np.asscalar(x[1:2,:].reshape(1,)), np.asscalar(x[2:3,:].reshape(1,))
#    c, s = np.cos(th), np.sin(th)
#    refs = [
#        plt.arrow(a - sc / 2 * c - delta * s, b - sc / 2 * s + delta * c, c * sc, s * sc, head_width=.05),
#        plt.arrow(a - sc / 2 * c + delta * s, b - sc / 2 * s - delta * c, c * sc, s * sc, head_width=.05)
#    ]
#    return refs

def lines(x,u,T):    
    t = np.linspace(0,T[-1]*dt,T[-1]+1)
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,sharex=True)
    for k in range(int(n/2)):
        ax1.plot(t,x[k,:],label="x"+str(k+1))
        ax2.plot(t,x[int(n/2)+k,:],label="v"+str(k+1))
        ax3.plot(t[:-1],u[k,:],label="u"+str(k+1))
    for k in range(len(T)):
        ax1.axvline(T[k]*dt,color="g",linestyle="--")
        ax2.axvline(T[k]*dt,color="g",linestyle="--")
        ax3.axvline(T[k]*dt,color="g",linestyle="--")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig.suptitle("Parallel DDP")
    ax3.set_xlabel("Time (seconds)")
    plt.savefig("courbes.png",dpi=1000)

x,u = constrainedLQR(x,u,T)
x,u = constrainedLQR(x,u,T)
lines(x,u,T)
