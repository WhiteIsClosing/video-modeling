from pylab import *                     
from tm_bias import tmat_bias as tmat, print_aligned
from std import show

def sigmoid(x):        return 1./(1.+exp(-x))
def Rsigmoid(x):       return array(rand(*shape(x)) < sigmoid(x), dtype=float)

class seq_rbm_base: #random is irrelevant now.
    def __init__(self, s1, s2, vh, hh, vv, contraction_factor=1, init=True, random=True):
        self.c = tmat(s1,s2, vh, offset=0, contraction_factor=contraction_factor, random=random)  
        self.a = tmat(s1,s1, vv, offset=1, init_mat=init, random=True) 
        self.b = tmat(s2,s2, hh, offset=1, init_mat=init, random=True) 
        self.H = array([],'d') 
        self.V = array([],'d')
        self.vislin = 0

    def copy(self):
        t = self.__class__(self.c.s1, self.c.s2, self.c.m, self.a.m, self.b.m,
                           init=False)
        t.c, t.a, t.b = self.c.copy(), self.a.copy(), self.b.copy()
        t.H, t.V      = self.H.copy(), self.V.copy()
        t.vislin      = self.vislin
        return t

    def zero_copy(self):
        t = self.__class__(self.c.s1, self.c.s2, self.c.m, self.a.m, self.b.m,
                           init=False)
        t.c, t.a, t.b = self.c.zero_copy(), self.a.zero_copy(), self.b.zero_copy()
        t.H, t.V      = 0*self.H.copy(), 0*self.V.copy()
        t.vislin      = self.vislin
        return t

    def grab_state(self, s):
        self.V = s.V.copy() 
        self.H = s.H.copy()
        self.c.vT = self.a.vT = self.a.hT = len(self.V)
        self.c.hT = self.b.vT = self.b.hT = len(self.H)

    def __add__ (self, t):
        s = self.copy()
        s.a += t.a; s.b += t.b;  s.c += t.c
        return s
    def __iadd__(self, t):
        self.a += t.a; self.b += t.b; self.c += t.c
        return self
    def __sub__ (self, t):
        s = self.copy()
        s.a -= t.a; s.b -= t.b;  s.c -= t.c
        return s
    def __isub__(self, t):
        self.a -= t.a;   self.b -= t.b; self.c -= t.c
        return self
    def __rmul__(self, c):
        s = self.copy();
        s.a=c * s.a; s.b=c * s.b; s.c=c * s.c
        return s
    def outp(self, V1=None, H1=None, V2=None, H2=None):
        if V1==None: V1=self.V
        if H1==None: H1=self.H
        if V2==None: V2=V1
        if H2==None: H2=H1

        s = self.zero_copy()
        s.c = self.c.outp(V2, H1)
        s.b = self.b.outp(H2, H1)
        s.a = self.a.outp(V2, V1)
        return s

#################################################################################

def stochastic(X): return array(apply(rand,shape(X))<X, 'd')
def id(x): return x
#################################################################################
class arbm(seq_rbm_base):
    para_gibbs = 2
    def infer(self,mf=1):
        sig = [Rsigmoid, sigmoid][mf]        
        BOTTOM_UP =  self.c * self.V 
        self.H = sig(BOTTOM_UP)
        if self.b.m>0:
            for g in xrange(arbm.para_gibbs): 
                m = self.b.m
                for mu in range(m+1):
                    INP_1 = self.b * self.H
                    INP_2 = self.b.tr() * self.H
                
                    H_probs = sig(BOTTOM_UP + INP_1 + INP_2)
                    self.H[mu::m+1] = H_probs[mu::m+1]

    def recon(self, mf=1):
        sig = [Rsigmoid, sigmoid][mf]        
        sig = [sig, id][self.vislin]

        TOP_DOWN = self.c.tr() * self.H 
        self.V = sig(TOP_DOWN)
        if self.a.m>0:
            for g in xrange(arbm.para_gibbs): 
                m = self.a.m
                for mu in range(m+1):
                    INP_1 = self.a * self.V
                    INP_2 = self.a.tr() * self.V  # the order of INP_1 and INP_2 matters!
                    # a bit inefficient but good enough.
                    V_probs = sig(TOP_DOWN  + INP_1 + INP_2)
                    self.V[mu::m+1] = V_probs[mu::m+1]

    def d_learn_up(self, mf=1):
        d_pos = self.outp()
        self.infer(mf)
        d_neg = self.outp()
        return d_pos - d_neg

    def d_learn_down(self, mf=1):
        d_pos = self.outp()
        self.recon(mf)
        d_neg = self.outp()
        return d_pos - d_neg

    def CDWS_recon(self, G=3):
        for g in xrange(G):
            self.infer(mf=0)
            self.recon(mf=0)

    def CD(self,G=2):
        d_pos = self.outp()
        for g in xrange(G):
            self.H[:]=stochastic(self.H[:])
            self.recon(mf=1)
            self.infer(mf=0)
        d_neg = self.outp()
        return d_pos-d_neg

    def gen(self, G=50):
        for g in range(G):
            self.recon(mf=0)
            self.infer(mf=1) #play around with this parameter.
        self.recon(mf=1)
        return self.V

############################################################################
############################################################################
############################################################################
## auxiliary function for TRBM

def forward_sigmoidal_pass(m,BOTTOM_UP,sig):
    T = len(BOTTOM_UP)
    assert(m.s1==m.s2) 
    H   = zeros((T, m.s1),'d') 
    for t in range(T):
        H[t]=sig(BOTTOM_UP[t] + m.local_mul(H, t))
    return H


class trbm(seq_rbm_base):
    def infer(self, mf=1): 
        sig = [Rsigmoid, sigmoid][mf]        
        BOTTOM_UP = self.c * self.V
        if self.b.m==0:  self.H= sig(BOTTOM_UP)
        else:            self.H= forward_sigmoidal_pass(self.b, BOTTOM_UP, sig)

    def recon(self, mf=1):

        sig = [Rsigmoid, sigmoid][mf]        
        sig = [sig, id][self.vislin]
        #assert(sig==1)
        TOP_DOWN =  dot(self.H, self.c.w[0].transpose())  + self.c.b_dn[0]
        if self.a.m==0:  self.V= sig(TOP_DOWN)
        else:            self.V= forward_sigmoidal_pass(self.a, TOP_DOWN, sig)

    def d_learn_up(self, mf=1):
        sig = [Rsigmoid, sigmoid][mf]
        H_pred = sig(self.c * self.V + self.b * self.H) 
        return self.outp(H1=self.H - H_pred, H2=self.H) 

    def d_learn_down(self, mf=1):
        sig = [Rsigmoid, sigmoid][mf]
        sig = [sig, id][self.vislin]

        V_pred = sig(dot(self.H, self.c.w[0].transpose()) + self.c.b_dn[0] + self.a * self.V)
        d = self.zero_copy()
        d.c.w[0]    = dot((self.V - V_pred).transpose(), self.H)
        d.c.b_dn[0] = sum(self.V - V_pred, 0)
        d.a   = d.a.outp(self.V, self.V - V_pred)
        return d

    def CD(self, G=1, mf=1):
        sig = [Rsigmoid, sigmoid][mf]
        sig = [sig, id][self.vislin]

        V_old, H_old = self.V.copy(), self.H.copy()
        d_pos = self.outp()
        self.c.offset=1
        b_h  = self.c * V_old + self.b * H_old 
        b_v  = self.a * V_old
        for g in xrange(G):
            self.H[:] = stochastic(self.H[:]) 
            self.V[:] = sig  (dot(self.H, self.c.w[0].transpose()) + self.c.b_dn[0] + b_v)
            self.H[:] = sigmoid (dot(self.V, self.c.w[0]) + self.c.b_up[0] + b_h)
        d_neg          = self.outp(V2=V_old, H2=H_old)
        d_neg.c.w[0]   = dot(self.V.transpose(), self.H) #the RBM weights.
        d_neg.c.b_up[0]= sum(self.H)
        d_neg.c.b_dn[0]= sum(self.V)
        self.c.offset=0
        return d_pos - d_neg

    def CDWS_recon(self, V=None, H=None, G=2, mf=0):
        if V==None: V=self.V
        if H==None: H=self.H
        T = len(V) 
        sig = [Rsigmoid, sigmoid][mf]
        sig = [sig, id][self.vislin]
        for t in xrange(T): #make a modification to the data.
            bv  = self.a.local_mul(V, t) 
            bh  = self.b.local_mul(H, t) + self.c.local_mul(V, t)            
            for g in xrange(G):
                H[t] = Rsigmoid(bh + dot(V[t], self.c.w[0]) + self.c.b_up[0]) 
                V[t] = sig(bv + dot(H[t], self.c.w[0].transpose()) + self.c.b_dn[0])

        
    def gen(self, T=50):
        G = 60
        V=self.V.copy()
        H=self.H.copy()
        self.CDWS_recon(V,H,G,mf=1)
        return V


    
