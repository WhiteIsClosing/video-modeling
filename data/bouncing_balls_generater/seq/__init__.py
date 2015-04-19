import seq.crbms_trainers5 as ct


def copy_trainer5(t):
    t1  = ct.trainer(*([None]*9))
    t1.nns = [x.copy() for x in t.nns]
    return t1

# this procedure is neccesary for the following reason: when we save and then
# load a file, sometimes python crashes. So what you do is this:
# you load a file, f=copy_trainer

