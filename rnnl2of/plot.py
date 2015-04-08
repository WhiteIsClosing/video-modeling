# plot arrays and store the frames under specified path

import numpy
from scipy import misc
      
# plot 1 channel frame
## ignore this: the frames should be flattened before plotting
def plotFrames(frames_ori, image_shape, path, max_num):
  frames = frames_ori.flatten()
  frame_dim = image_shape[0] * image_shape[1] 
  frames_dim = frames.shape[0]
  num = frames_dim / frame_dim
  for i in range(min(num, max_num)):
    frame = frames[i*frame_dim:(i+1)*frame_dim, None]
    img = numpy.reshape(frame, image_shape)
    misc.imsave(path + str(i) + '.jpeg', img)

