import matplotlib
matplotlib.use('Agg')

from matplotlib.pylab import plt

from skimage.io import imread, imsave

whole_set = '2.png'
out_path  = '22.png'

data = imread(whole_set)

plt.imshow(data, cmap=plt.cm.seismic)
plt.axis('off')
plt.savefig(out_path, dpi=300)