import numpy
from scipy import misc

def rad_map(vec):
  '''
  Map 2D vector to rad \in [0, 2*\pi].
  '''
  if vec[1] == 0:
    rad = numpy.arctan(vec[0] / (vec[1]+1e-20))
  else:
    rad = numpy.arctan(vec[0] / vec[1])

  if vec[0] < 0 and vec[1] >= 0:
    rad += 2*numpy.pi
  elif vec[0] < 0 and vec[1] < 0:
    rad += numpy.pi
  elif vec[0] >= 0 and vec[1] < 0:
    rad += numpy.pi

  return rad


def color_map(vec, maxi, mini):
  '''
  Map 2D vector to RGB color.
  '''
  length = numpy.sqrt(vec[0]**2 + vec[1]**2)

  if (length <= mini):
    scale = 0.
  elif (length >= maxi):
    scale = 255.
  else:
    scale = 255 * (length - mini) / maxi

  rad = rad_map((vec[0], -vec[1]))
  bnd = numpy.pi*2/3
  if rad >= 0 and rad < bnd:
    theta = (rad - 0.) / bnd
    r = scale * numpy.cos(theta)
    g = scale * numpy.sin(theta)
    b = 0.
  elif rad >= bnd and rad < 2*bnd:
    theta = (rad - bnd) / bnd
    r = 0.
    g = scale * numpy.cos(theta)
    b = scale * numpy.sin(theta)
  else: 
    theta = (rad - bnd) / bnd
    r = scale * numpy.sin(theta)
    g = 0.
    b = scale * numpy.cos(theta)

  return (r, g, b)

def xy2rgb(framex, framey, maxi, mini):
  '''
  Plot x and y component of the frame into a RGB frame according to the color 
  map.
  '''
  (m, n) = framex.shape
  img = numpy.zeros((m, n, 3))
  for x in range(m):
    for y in range(n):
      rgb = color_map((framex[x, y], framey[x, y]), maxi, mini)
      img[x, y, :] = rgb
      # img = misc.toimage(img, cmin=0, cmax=255, mode='P')
  return img

