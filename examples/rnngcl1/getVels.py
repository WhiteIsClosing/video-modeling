
def getVels(x, gc):
# grammar-cell velocity mapping unit
  frame_left = T.matrix(name='frame_left')
  frame_right = T.matrix(name='frame_right')
  factor_left = T.dot(frame_left, gc.wxf_left)
  factor_right = T.dot(frame_right, gc.wxf_right)
  vel_ = T.nnet.sigmoid(T.dot(factor_left*factor_right, gc.wv)+gc.bv)
  getVel = theano.function([frame_left, frame_right], vel_)

  x_left = T.concatenate((T.zeros((1, x.shape[1])), x[:-1]), axis=0)
  x_right = x

  vels = numpy.zeros((x.shape[0], numvel_))
  for t in range(x.shape[0]):
    vel = getVel(x_left[t, :], x_right[t, :])
    vels[t, :] = vel
