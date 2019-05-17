#matplotlib inline
#config InlineBackend.figure_format= 'retina'
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical
batch_size = 128
num_classes=10
epochs= 20
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print('training data shape:', x_train.shape,y_train.shape)
print('testing data shape:', x_test.shape,y_test.shape)
print('-'*55)
classes= np.unique(y_train)
nClasses =len (classes)
print('total number of outputs:',nClasses)
print('output classes         :',classes)
plt.figure(figsize=[10,5])
plt.subplot(121)
plt.imshow(x_train[0,:,:],cmap='gray')

plt.title("Ground Truth : {}".format(y_train[0]))

plt.subplot(122)
plt.imshow(x_test[0,:,:],cmap='gray')
plt.title("ground truth : {}".format(y_test[0]))
plt.show()
