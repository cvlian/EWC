import os
import gzip
import pickle
from utils import *

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

class Dataset:

    def getRawData(self):
        return self.x, self.y

    def showSamples(self, nrows, ncols):
        """
        Plot nrows x ncols images
        """
        fig, axes = plt.subplots(nrows, ncols)
        for i, ax in enumerate(axes.flat): 
            ax.imshow(self.x[i,:,:,0])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(np.argmax(self.y[i]))
        
        plt.show()

class MNISTdata(Dataset):

    def __init__(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.x = np.reshape(mnist.train.images, [-1, 28, 28, 1])
        self.y = mnist.train.labels

    def getTask(self):
        samples = np.random.choice(range(self.x.shape[0]), 25000, replace=False)

        x_train, x_val, y_train, y_val = train_test_split(self.x[samples], self.y[samples], test_size=0.05)
        self.task = {'name':'mnist', 'x_train':x_train, 'x_val':x_val, 'y_train':y_train, 'y_val':y_val}

        print("MNIST : Training Set", x_train.shape)
        print("MNIST : Test Set", x_val.shape)

        # Calculate the total number of images
        num_images = x_train.shape[0] + x_val.shape[0]
        print("MNIST : Total Number of Images", num_images)

        return self.task
    
class QMNISTdata(Dataset):

    def __init__(self):
        image_size = 28
        num_images = 60000

        f = gzip.open('../dataset/qmnist-images.gz','r')
        f.read(16)

        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        x = data.reshape(num_images, image_size, image_size, 1)

        f = gzip.open('../dataset/qmnist-labels.gz','r')
        f.read(8)

        buf = f.read(num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        y = data.reshape(num_images, 1)

        # Fit the OneHotEncoder
        enc = OneHotEncoder().fit(y.reshape(-1, 1))

        # Transform the label values to a one-hot-encoding scheme
        y = enc.transform(y.reshape(-1, 1)).toarray()

        self.x = x
        self.y = y

    def getTask(self):
        samples = np.random.choice(range(self.x.shape[0]), 10000, replace=False)

        x_train, x_val, y_train, y_val = train_test_split(self.x[samples], self.y[samples], test_size=0.1)
        self.task = {'name':"qmnist", 'x_train':x_train, 'x_val':x_val, 'y_train':y_train, 'y_val':y_val}

        print("QMNIST : Training Set", x_train.shape)
        print("QMNIST : Test Set", x_val.shape)

        # Calculate the total number of images
        num_images = x_train.shape[0] + x_val.shape[0]
        print("QMNIST : Total Number of Images", num_images)

        return self.task

class MNIST_Mdata(Dataset):

    def __init__(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        mnistm = pickle.load(open('../dataset/mnistm_data.pkl', 'rb'), encoding='latin1')

        x = np.reshape(mnistm['train'], [-1, 28, 28, 3])
        x = rgb2gray(x).astype(np.float32)

        self.x = x
        self.y = mnist.train.labels

    def getTask(self):
        samples = np.random.choice(range(self.x.shape[0]), 10000, replace=False)

        x_train, x_val, y_train, y_val = train_test_split(self.x[samples], self.y[samples], test_size=0.1)
        self.task = {'name':"mnist-m", 'x_train':x_train, 'x_val':x_val, 'y_train':y_train, 'y_val':y_val}

        print("MNIST-M : Training Set", x_train.shape)
        print("MNIST-M : Test Set", x_val.shape)

        # Calculate the total number of images
        num_images = x_train.shape[0] + x_val.shape[0]
        print("MNIST-M : Total Number of Images", num_images)

        return self.task
