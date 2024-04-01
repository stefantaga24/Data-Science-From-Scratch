import mnist
from deep_learning import shape
mnist.temporary_dir  = lambda :'/tmp'

train_images = mnist.train_images().tolist()
train_labels = mnist.train_labels().tolist()

assert shape(train_images) ==[60000, 28,28]
assert shape(train_labels) ==[60000]

import matplotlib.pyplot as plt

fig,ax = plt.subplots(10,10)

for i in range(10):
    for j in range(10):
        ax[i][j].imshow(train_images[i*10+j],cmap ='Greys')
        ax[i][j].xaxis.set_visible(False)
        ax[i][j].yaxis.set_visible(False)

plt.show()