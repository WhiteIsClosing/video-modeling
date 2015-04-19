import numpy
from scipy import misc
from color_map import *
      
def plotFrames(frames_ori, image_shape, path, max_num):
  '''
  Plot 1 channel frame.
  '''
  frames = frames_ori.flatten()
  frame_dim = image_shape[0] * image_shape[1] 
  frames_dim = frames.shape[0]
  num = frames_dim / frame_dim
  for i in range(min(num, max_num)):
    frame = frames[i*frame_dim:(i+1)*frame_dim, None]
    img = numpy.reshape(frame, image_shape)
    misc.imsave(path + str(i) + '.jpeg', img)

def plotFramesNoScale(frames_ori, image_shape, path, max_num):
  '''
  Plot 1 channel frame without scaling.
  '''
  frames = frames_ori.flatten()
  frame_dim = image_shape[0] * image_shape[1] 
  frames_dim = frames.shape[0]
  num = frames_dim / frame_dim
  for i in range(min(num, max_num)):
    frame = frames[i*frame_dim:(i+1)*frame_dim, None]
    img = numpy.reshape(frame, image_shape)
    #misc.imsave(path + str(i) + '.jpeg', img)
    im = misc.toimage(img, cmin=0, cmax=255)  #ToDo
    im.save(path + str(i) + '.jpeg')

def plotOFs(ofx_ori, ofy_ori, maxi, mini, image_shape, path, max_num):
  '''
  Plot colored optical flow
  '''
  ofxs = ofx_ori.flatten()
  ofys = ofy_ori.flatten()
  frame_dim = image_shape[0] * image_shape[1] 
  frames_dim = ofxs.shape[0]

  assert ofxs.shape[0] == ofys.shape[0]
  # if (ofxs.shape[0] != ofys.shape[0]):
  #   print 'ERROR! ofxs.shape[0] != ofys.shape[0]'

  num = frames_dim / frame_dim
  for i in range(min(num, max_num)):
    ofx = ofxs[i*frame_dim:(i+1)*frame_dim, None]
    ofy = ofys[i*frame_dim:(i+1)*frame_dim, None]

    of = xy2rgb(ofx.reshape(image_shape), ofy.reshape(image_shape), maxi, mini)
    im = misc.toimage(of, cmin=0, cmax=255) # has to be ploted without scaling
    im.save(path + str(i) + '.jpeg')
