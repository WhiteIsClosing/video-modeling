from pylab import *
from data_dynamic import bounce_vec
import seq
# res is the only variable!
res, nballs, radius = 20, 2, 2
balls =   lambda : bounce_vec (res, n=nballs, r=[radius]*nballs, T=100)
from std import *

#################################### Getting the denoising images ####

import noise.time_noise as tn

def get_clean_image(self,D,l):
    for i in xrange(l):
        self.nns[i].V=D.copy()   
        self.nns[i].infer(mf=1)
        D = self.nns[i].H.copy()

    for i in reversed(xrange(l)):
        self.nns[i].H=D.copy() 
        self.nns[i].recon(mf=1)  
        D = self.nns[i].V.copy()
    return D

def make_image_comparison(t, data_getter=balls):
    """
    input: the trainer. output: an image analogous to the one in the paper.
    note that these models were not trained to denoise, and the fact that
    they can is merely 'nice'.
    """
    D1 = data_getter()
    D2 = tn.noise(D1, 0.004, temp=6) 
    D3 = get_clean_image(t,D2,1)
    D4 = get_clean_image(t,D2,2)
    r  = range(10,20)
    return concatenate(map(print_seq,[D1[r],D2[r],D3[r],D4[r]]))
                        
