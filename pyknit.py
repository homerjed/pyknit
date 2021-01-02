import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as plticker
from matplotlib.colors import ListedColormap
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np 
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('path',
                    help='name of image file (in current directory).')
parser.add_argument('n',
                    help='number of colours of wool required.')
parser.add_argument('d',
                    help='multiple to downsample image by.')                    
args = parser.parse_args()

n = args.n
k = args.d 
img = cv2.imread(os.path.join(os.getcwd(), args.path), cv2.IMREAD_GRAYSCALE)
img = squarify(img)

def squarify(img):
  # round smallest dim to max dim length 
  square = np.zeros((np.max(img.shape), np.max(img.shape)))
  if img.shape[0] < img.shape[1]:
    square[:-1,:] = img
    return square
  elif img.shape[1] < img.shape[0]:
    square[:,:-1] = img
    return square
  else:
    return img

def downsample(x, k=k):
  # average pooling
  dx = x.reshape((x.shape[0] // k, k,
                  x.shape[1] // k, k))
  return dx.mean(axis=(1,3)) 

def norm(x):
  # [0,1] normalisation
  return (x - x.min()) / (x.max() - x.min())

def density_partition(x, n=n):
  # normalise for safe partitioning in cond_
  x, y = norm(x), np.zeros_like(x)
  fills = [_ / n for _ in range(n)]

  for _ in range(n):
    cond_ = (_ / n < x) & (x < (_ + 1) / n)
    ix_ = np.argwhere(cond_)

    y[ix_[:,0], ix_[:,1]] = fills[_] #x[ix_[:,0], ix_[:,1]]
  return y

def make_cmap(rgb):
  r, g, b = rgb
  N = 256
  vals = np.ones((N, 4))
  # vals[:, 0] = np.linspace(r / 256, 1, N)
  # vals[:, 1] = np.linspace(g / 256, 1, N)
  i = np.random.randint(0,2)
  vals[:, i] = np.linspace(b / 256.0, 1.0, N)
  map = ListedColormap(vals)
  rgba = (1-0.99,1-0.99,1-0.99)
  # map.set_over(rgba, alpha=0.0)
  map.set_under(rgba, alpha=0.0)
  return map

colours = ['#A24343', '#DACBC1', '#330033', 
           '#861186', '#311E5E', '#C6CE92']
colours = ['red', '#000000','#444444', '#666666',
            '#ffffff', 'blue', 'orange']


def random_colour_set(n=n):
  r = lambda: np.random.randint(0,255)
  return ['#%02X%02X%02X' % (r(),r(),r()) for _ in range(n)]

def rgb_hex(hexes):
  # convert hex codes to rgb
  rgb = lambda hex : tuple(int(hex[_ : _ + 2], 16) for _ in (1, 3, 5))
  return [np.sum(np.square(rgb(hex))) for hex in hexes]

def discrete_cmap(colours=None):
  if colours == None:
    colours = random_colour_set()
    colours = [c for c,_ in sorted(zip(colours,rgb_hex(colours)))]

  boundaries = [_ / n for _ in range(n)] 
  cmap = colors.ListedColormap(colours)
  norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
  return cmap

def make_cmap(rgb):
  r, g, b = rgb
  N = 256
  vals = np.ones((N, 4))
  # vals[:, 0] = np.linspace(r / 256, 1, N)
  # vals[:, 1] = np.linspace(g / 256, 1, N)
  i = np.random.randint(0,2)
  vals[:, i] = np.linspace(b / 256.0, 1.0, N)
  map = ListedColormap(vals)
  rgba = (1-0.99,1-0.99,1-0.99)
  # map.set_over(rgba, alpha=0.0)
  map.set_under(rgba, alpha=0.0)
  return map

def separate_partition(x, i):
  y = np.zeros_like(x)
  y[:,:] = 1e-6

  cond_ = (i / n < x) & (x < (i + 1) / n)
  ix_ = np.argwhere(cond_)

  y[ix_[:,0], ix_[:,1]] = x[ix_[:,0], ix_[:,1]]
  return y

comps = False
if comps:
  fig, axs = plt.subplots(1, n, figsize=(6 * n, 6), dpi=100)
  ix = [_ for _ in range(n)]
  for i,ax in zip(ix, axs):
    ax.set_title("%.2f - %.2f" % (i / n, (i + 1) / n))
    ax.imshow(
        separate_partition(norm(X), i),
        cmap=make_cmap(np.random.randint(10,236,size=3)),
        interpolation='none', 
        clim=[1e-4, 1]
        )
    ax.axis('off')
  plt.show()

part_img = density_partition(downsample(img))

hist = False
if hist:
  fig, ax = plt.subplots(figsize=(12,2), dpi=100)
  ax.patch.set_facecolor((0,0,0,0))
  bins = [_ / n for _ in range(n + 1)]
  ax.hist(norm(part_img.flatten()), histtype='bar', density=True, 
          label='d_image', alpha=0.7, bins=bins, color='rebeccapurple')
  ax.hist(norm(img.flatten()), histtype='step', density=True, 
          label='image', 
          bins=[0.1 * _ / n for _ in range(10 * n)],
          color='orchid')
  ax.legend(loc='upper left')

plot_knitted = True
if plot_knitted:
  fig, (ax1, ax2) = pt.subplots(1, 2, figsize=(12,12), dpi=100)

  q = ax1.imshow(img, cmap='gray')
  divider = make_axes_locatable(ax1)
  cax1 = divider.append_axes('right', size='5%', pad=0.05)
  fig.colorbar(q, 
              cax=cax1,
              orientation='vertical', ax=ax1)
  ax1.axis('off')

  p = ax2.imshow(part_img, cmap=discrete_cmap(colours=None))
  ax2.axis('off')
  divider = make_axes_locatable(ax2)
  cax2 = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar(p, 
              cax=cax2, 
              orientation='vertical', ax=ax2)
  plt.show()
