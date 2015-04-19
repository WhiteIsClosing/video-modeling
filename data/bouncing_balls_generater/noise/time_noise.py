from pylab import *
from scipy.signal import convolve2d as conv2

ker1   = array([[.1, .5, .5, .1],
                [.5, 1,   1, .5],
                [.5, 1,   1, .5],
                [.1, .5, .5, .1]])

ker2   = array([[.4, .7, .8, .7, .4],
                [.7, 1, 1, 1, .7],
                [.8, 1, 1, 1, .8],
                [.7, 1, 1, 1, .7],
                [.1, .5, .8, .7, .4]])

def noise(x, freq, temp=0, ker=ker2):
    s1 = int(sqrt(shape(x)[-1]))
    
    T  = shape(x)[0]
    shape_x = (T, s1, s1)

    y  = zeros(shape_x,'d')

    moi = array(rand(T+temp,s1,s1)<freq,'d')
    noi = zeros(shape_x)

    for t in range(temp+1):
        noi+=moi[t:t+T]
    # python isn't too shabby. awesome.
    noi.putmask(1, noi>1)
    

    for t in range(T):
        y[t,:,:]  = conv2(noi[t,:,:], ker, 'same')

    y = y.reshape(shape(x))

    y+=x #x is the video to noise, after all :)

    y.putmask(1, y>1)

    return y 
    

