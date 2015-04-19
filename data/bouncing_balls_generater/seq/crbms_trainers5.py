from crbms2 import *

normalize  = lambda X: X/abs(X).max()
hcat = lambda V,H: array(list(V.transpose())+list(H.transpose())).transpose()
vcat = lambda V,H: array(list(V)+list(H))
def npa(x): return normalize(print_aligned(x))
def norm(x): return sum(x.ravel()**2)
from std import save, show
class trainer:

    def __init__(self, nns, LR, momentum, accel_rbm, accel_ws, 
                 maxepoch, data_getter, saving_times,   save_path):
        if nns==None:
            return

        self.done_greedy = False
        self.nns = nns
        self.nns_recog = None 
        self.data_getter = data_getter
        self.momentum = momentum
        self.LR       = LR
        self.Orig_LR  = LR
        self.accel_epochs_rbm = accel_rbm
        self.accel_epochs_ws  = accel_ws
        self.save_path = save_path
        self.saving_times = saving_times
        self.print_freq = 100
        self.epoch = 1
        self.maxepoch = maxepoch
        self.layer = 0 
        self.v = [0*nn for nn in self.nns] 

    def data_getter_layer(self, layer):
        D = self.data_getter()
        for i in xrange(layer):
            self.nns[i].V = D.copy()
            self.nns[i].H = zeros(shape(self.nns[i].c * D),'d') 
            self.nns[i].infer()
            D = self.nns[i].H.copy()
        return D

    def gen_layer(self, layer=None):
        if layer==None: layer=len(self.nns)
        D = self.nns[layer].gen()
        for i in reversed(xrange(layer)):
            self.nns[i].H = D
            self.nns[i].recon(mf=1)
            D = self.nns[i].V
        return D
    
    def train(self, nn_num=0):
        nn = self.nns[nn_num]
        v  = self.v[nn_num]
        nn.H = 0*(nn.c * self.data_getter_layer(nn_num))

        for self.epoch in xrange(self.epoch, self.maxepoch):
            # decide whether or not to save.
            if self.epoch in self.saving_times:
                tmp = self.data_getter
                self.data_getter = None #this hack is needed because pickle
                                        #cannot save a function.
                save_str= self.save_path + 'layer_is_'+str(nn_num).zfill(2)+'_and_epoch_is_'+str(self.epoch).zfill(7)
                print save_str
                save(self, save_str)
                self.data_getter = tmp 
                
            try:
                self.LR *= self.accel_epochs_rbm[self.epoch]
            except: None

            nn.V = self.data_getter_layer(nn_num)
            self.T = len(nn.V)
            nn.infer()

            d         = nn.CD()
            d.a  = .1 * d.a #autoregressive connections learn more slowly.
            d.b  =  1 * d.b 
            self.d = d
            v = self.momentum * v + d
            nn += self.LR  * v

            c = nn.c; b = nn.b
            print ('layer=%d, epoch=%d, LR=%f, m=%f, |d.c.w|=%f, |c.w|=%f, |d.b.w|=%f, |b.w|=%f' % 
                   ((self.layer, self.epoch, self.LR, self.momentum,)+
                    tuple(norm(x) for x in [d.c.w, c.w, d.b.w, b.w])))


            try: #what if for some reason we cannot plot?
                if self.epoch % self.print_freq == 0 and nn_num==0:
                    if nn.c.m>0:
                        figure(1)
                        show(vcat(hcat(npa(nn.c.w[0]),npa(nn.c.w[1])),
                                  hcat(npa(d.c.w[0]) ,npa(d.c.w[1]))))
                    else:
                        figure(1)
                        show(vcat(npa(nn.c.w[0]),npa(d.c.w[0])))
                    if nn.a.m>0:
                        figure(2)
                        show(hcat(npa(nn.a.w[1]), npa(d.a.w[1])))
                        
            except: None

    def greedy_train(self): #pausable
        for self.layer in xrange(self.layer, len(self.nns)):
            self.train(self.layer)
            self.epoch=1
            self.LR = self.Orig_LR 
        self.done_greedy = True

