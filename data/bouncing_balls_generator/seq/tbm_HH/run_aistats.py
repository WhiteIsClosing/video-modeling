from pylab import *
from seq.crbms_trainers5 import *
from seq.data_dynamic import bounce_vec
res, nballs, radius = 20, 2, 2
balls =   lambda : bounce_vec (res, n=nballs, r=[radius]*nballs, T=100)

fun = balls

y = [None]
def doit(x):
    y[0]=x
    x.greedy_train()

def resume():
    y[0].greedy_train()


numhid = 200

M   = 4

accel_rbm = { 100:2, 200:2, 500:2, 1000:2}
accel_ws  = {}


K   = 1
t_nn1  = K*(1./res**2/M)*trbm(res**2, numhid,   vh=M, hh=0, vv=M, random=True)
t_nn2  = K*(1./res**2/M)*trbm(numhid, 2*numhid, vh=M, hh=0, vv=M, random=True)
t_nn3  = K*(1./res**2/M)*trbm(2*numhid, numhid, vh=M, hh=0, vv=M, random=True)
save_path = 'seq/tbm_HH/run_aistats_'



t_t2 = trainer(nns=[t_nn1, t_nn2, t_nn3], LR=0.00005, momentum=.9, 
               maxepoch=10000,
               accel_ws=accel_ws, accel_rbm=accel_rbm,
               data_getter = fun, saving_times=[9999], save_path = save_path) 

t_t2.print_freq=20

try:    doit(t_t2)
except KeyboardInterrupt: None
except: raise










