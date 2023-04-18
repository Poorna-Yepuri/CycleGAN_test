###########################################

# Use the saved cyclegan models for image translation
from os import listdir
import cv2
from numpy import asarray
from numpy import vstack
from instancenormalization import InstanceNormalization  
from keras.models import load_model
from matplotlib import pyplot
from numpy.random import randint
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import resample

from Resize_Images import upscale, downsize


# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X

# plot the image, its translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
	images = vstack((imagesX, imagesY1, imagesY2))
	titles = ['Real', 'Generated', 'Reconstructed']
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, len(images), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# title
		pyplot.title(titles[i])
	pyplot.show()

# load dataset
A_data = resample(dataA, 
                 replace=False,     
                 n_samples=50,    
                 random_state=42) # reproducible results

B_data = resample(dataB, 
                 replace=False,     
                 n_samples=50,    
                 random_state=42) # reproducible results

A_data = (A_data - 512) / 512
B_data = (B_data - 512) / 512


# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('models/g_model_AtoB_010000.h5', cust)
model_BtoA = load_model('models/g_model_BtoA_010000.h5', cust)

def example_A_B():
    # plot A->B->A (Painting to Photo to Painting)
    A_real = select_sample(A_data, 1)
    B_generated  = model_AtoB.predict(A_real)
    A_reconstructed = model_BtoA.predict(B_generated)
    show_plot(A_real, B_generated, A_reconstructed)

# plot B->A->B (Photo to Painting to Photo)
def example_B_A():
    B_real = select_sample(B_data, 1)
    A_generated  = model_BtoA.predict(B_real)
    B_reconstructed = model_AtoB.predict(A_generated)
    show_plot(B_real, A_generated, B_reconstructed)

##########################
#Load a single custom image
def life_of_photo(infer_img):
    test_image = load_img(infer_img)
    test_image = img_to_array(test_image)
    test_image_input = np.array([test_image]) 
    test_image_input = (test_image_input - 512) / 512

    # plot B->A->B (Photo to Painting to Photo)
    painting_generated  = model_BtoA.predict(test_image_input)
    photo_reconstructed = model_AtoB.predict(painting_generated)
    show_plot(test_image_input, painting_generated, photo_reconstructed)

def infer_img(img_path):
    img = cv2.imread(img_path)
    test_img_shape = img.shape
    if test_img_shape[1] < 1024 and test_img_shape[0] < 1024:
        inp_img = upscale(img, test_img_shape)
    else:
        inp_img = downsize(img, 1024)
    
    test_image = img_to_array(inp_img)
    test_image_input = np.array([test_image])  # Convert single image to a batch.
    test_image_input = (test_image_input - 127.5) / 127.5

    # plot B->A->B (Photo to Monet to Photo)
    GAN_generated  = model_BtoA.predict(test_image_input)
    photo_reconstructed = model_AtoB.predict(GAN_generated)
    show_plot(test_image_input, GAN_generated, photo_reconstructed)

    
