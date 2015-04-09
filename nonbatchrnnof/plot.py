# plot arrays and store the frames under specified path

import numpy
from scipy import misc
from color_map import *
      
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

def plotFrames2(frames_ori, image_shape, path, max_num):
  frames = frames_ori.flatten()
  frame_dim = image_shape[0] * image_shape[1] 
  frames_dim = frames.shape[0]
  num = frames_dim / frame_dim
  for i in range(min(num, max_num)):
    frame = frames[i*frame_dim:(i+1)*frame_dim, None]
    img = numpy.reshape(frame, image_shape)
    #misc.imsave(path + str(i) + '.jpeg', img)
    im = misc.toimage(img, cmin=0, cmax=255)
    im.save(path + str(i) + '.jpeg')

# plot colored optical flow
def plotOFs(ofx_ori, ofy_ori, maxi, mini, image_shape, path, max_num):
  ofxs = ofx_ori.flatten()
  ofys = ofy_ori.flatten()
  frame_dim = image_shape[0] * image_shape[1] 
  frames_dim = ofxs.shape[0]

  if (ofxs.shape[0] != ofys.shape[0]):
    print 'ERROR! ofxs.shape[0] != ofys.shape[0]'

  num = frames_dim / frame_dim
  for i in range(min(num, max_num)):
    ofx = ofxs[i*frame_dim:(i+1)*frame_dim, None]
    ofy = ofys[i*frame_dim:(i+1)*frame_dim, None]

    of = xy2rgb(ofx.reshape(image_shape), ofy.reshape(image_shape), maxi, mini)
    im = misc.toimage(of, cmin=0, cmax=255)
    im.save(path + str(i) + '.jpeg')
