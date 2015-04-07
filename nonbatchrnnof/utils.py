import numpy
from scipy import misc

class LogInfo(object):
  def __init__(self, log_name, verbose = 1):
    self.flog = open(log_name, 'w')
    self.verbose = verbose

  def __del__(self):
    self.flog.close()
    print 'Finished logging.'
  
  def mark(self, content):
    if (self.verbose == 1):
      print content
      self.flog.write(content + '\n')
      self.flog.flush()
      
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

