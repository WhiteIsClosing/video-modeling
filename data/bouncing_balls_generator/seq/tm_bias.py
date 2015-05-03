from pylab import *
def expand(V,contraction_factor,vT):
    A = zeros((vT,)+shape(V)[1:],'d')
    A[::contraction_factor]=V
    return A
class tmat_bias:
    def __init__(self, s1, s2, m, init_mat=True, is_tr=False, random=True,
                 scale=0.1, offset=0, contraction_factor=1):

        self.s1, self.s2, self.m, self.is_tr, self.offset = int(s1), int(s2), m, is_tr, offset
        self.contraction_factor = contraction_factor
        self.vT = 0; self.hT = 0
        if init_mat: 
            if random:
                self.w    = randn(m+1,s1,s2)*scale
                self.b_up = randn(m+1,s2)*scale
                self.b_dn = randn(m+1,s1)*scale
            else:
                self.w    = zeros((m+1,s1, s2),'d')
                self.b_up = zeros((m+1,s2),'d')
                self.b_dn = zeros((m+1,s1),'d')

    def copy(self):
        m = tmat_bias(self.s1, self.s2, self.m, init_mat=False, is_tr=self.is_tr,
                      offset=self.offset, contraction_factor=self.contraction_factor)
        m.vT, m.hT = self.vT, self.hT
        m.w    = self.w.copy()
        m.b_up = self.b_up.copy()
        m.b_dn = self.b_dn.copy()
        return m

    def zero_copy(self):
        m =  tmat_bias(self.s1, self.s2, self.m, random=False, is_tr=self.is_tr, offset=self.offset,
                       contraction_factor=self.contraction_factor)
        m.vT, m.hT = self.vT, self.hT
        return m

    def __rmul__(self, c):
        m=self.copy()
        m.w    *=c
        m.b_up *=c
        m.b_dn *=c
        return m
    def __add__ (self, s):
        m=self.copy()
        m.w    +=s.w
        m.b_up +=s.b_up
        m.b_dn +=s.b_dn
        return m
    def __sub__ (self, s):
        m=self.copy()
        m.w     -=s.w
        m.b_up  -=s.b_up
        m.b_dn  -=s.b_dn
        return m
    def __iadd__(self, m):
        self.w    += m.w
        self.b_up += m.b_up
        self.b_dn += m.b_dn
        return self
    def __isub__(self, m):
        self.w    -= m.w
        self.b_up -= m.b_up
        self.b_dn -= m.b_dn
        return self

    def __mul__ (self, V):
        s1, s2, m, contraction_factor = self.s1, self.s2, self.m, self.contraction_factor
        offset=self.offset
        T = len(V)
        #print T
        if not self.is_tr:
            self.vT = T
            H =  zeros((T, s2), 'd')
            if offset==0:
                H += dot(V,self.w[0]) 
                H += self.b_up[0] 
            for i in range(max(1, offset), m+1):
                H[i:] += dot( V[:-i], self.w[i]) 
                H[:i] += self.b_up[i]

            if self.contraction_factor!=1: #contract H.
                H = H[::self.contraction_factor]
            self.hT = len(H)
            return H
        else:
            # although its V, its actually "H", cause we multiply by W.transpose().
            if contraction_factor!=1: V=expand(V,contraction_factor,self.vT)
            U = zeros((self.vT, s1), 'd') 
            if offset==0:
                U += dot(V,self.w[0].transpose())
                U += self.b_dn[0]
            for i in range(max(1, offset), m+1):
                U[:-i] += dot(V[i:], self.w[i].transpose())
                U[-i:] += self.b_dn[i]
            return U

    def tr(self):
        m_tr = tmat_bias(self.s1, self.s2, self.m, init_mat=False, is_tr=not self.is_tr, offset=self.offset,
                         contraction_factor=self.contraction_factor)
        m_tr.hT   = self.hT; m_tr.vT = self.vT
        m_tr.w    = self.w
        m_tr.b_dn = self.b_dn
        m_tr.b_up = self.b_up
        return m_tr

    def outp(self, V, H):
        if self.is_tr: TMP=V.copy(); V=H.copy(); H=TMP; # swap V and H.
        if self.contraction_factor!=1: H=expand(H,self.contraction_factor,self.vT)
        
        m = self.zero_copy()
        assert(len(V)==len(H))
        if self.offset==0:
            m.w[0] =  dot(V.transpose(), H)
            m.b_up[0] = sum(H, 0)
            m.b_dn[0] = sum(V, 0)
        for i in range(max(1, self.offset), self.m+1):
            m.w[i]    = dot(V[:-i].transpose(), H[i:])
            m.b_up[i] = sum(H[:i] ,0)
            m.b_dn[i] = sum(V[-i:],0)
        return m

    # figure out how to implement contraction factor; not urgent, though.
    def local_mul(self, X, t):
        A = zeros(self.s2, 'd')
        if self.offset==0: A+= self.b_up[0] # we don't use the main bias unless this case is true.
        for i in range(max(1, self.offset), 1+min(t, self.m)):  
            A += dot(X[t-i],self.w[i])
        for i in range(1+t, 1+self.m):
            A += self.b_up[i]
        return A

    

def print_aligned(w):
    n1 = int(ceil(sqrt(shape(w)[1])))
    n2 = n1
    r1 = int(sqrt(shape(w)[0]))
    r2 = r1
    Z = zeros(((r1+1)*n1, (r1+1)*n2), 'd')
    i1, i2 = 0, 0
    for i1 in range(n1):
        for i2 in range(n2):
            i = i1*n2+i2
            if i>=shape(w)[1]: break
            Z[(r1+1)*i1:(r1+1)*(i1+1)-1, (r2+1)*i2:(r2+1)*(i2+1)-1] = w[:,i].reshape(r1,r2)
    return Z


