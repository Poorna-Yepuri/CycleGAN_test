###########################################

# Use the saved cyclegan models for image translation
from os import listdir
import cv2
from numpy import asarray
from numpy import vstack
from instancenormalization import InstanceNormalization  
from tensorflow.keras.models import load_model
import keras.backend as K
from matplotlib import pyplot
from numpy.random import randint
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.utils import load_img
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import resample

from Resize_Images import upscale, downsize


# # select a random sample of images from the dataset
# def select_sample(dataset, n_samples):
# 	# choose random instances
# 	ix = randint(0, dataset.shape[0], n_samples)
# 	# retrieve selected images
# 	X = dataset[ix]
# 	return X

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

# # load dataset
# A_data = resample(dataA, 
#                  replace=False,     
#                  n_samples=50,    
#                  random_state=42) # reproducible results

# B_data = resample(dataB, 
#                  replace=False,     
#                  n_samples=50,    
#                  random_state=42) # reproducible results

# A_data = (A_data - 512) / 512
# B_data = (B_data - 512) / 512


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
K.set_image_data_format('channels_last')
def life_of_photo(infer_image):  # share path of infering image
    test_image = load_img(infer_image)
    test_image = img_to_array(test_image)
    print(test_image.shape)
    if K.image_data_format() == 'channels_first':
        test_image = np.transpose(test_image, (1, 2, 0))
    test_image_input = np.array([test_image]) 
    test_image_input = (test_image_input - 127.5) / 127.5
    print(test_image_input.shape)

    # plot B->A->B (Photo to Painting to Photo)
    GAN_painting_generated  = model_BtoA.predict(test_image_input)
    photo_reconstructed = model_AtoB.predict(GAN_painting_generated)
    show_plot(test_image_input, GAN_painting_generated, photo_reconstructed)

    generated_img_1 = 0.5 * GAN_painting_generated + 0.5
    print(generated_img_1.shape)
    generated_img = array_to_img(generated_img_1[0])
    return generated_img

def infer_img(img_path):
    img = cv2.imread(img_path)
    global test_img_shape, inp_img
    test_img_shape = img.shape
    if test_img_shape[1] < 256 and test_img_shape[0] < 256:
        inp_img = upscale(img, (256,256))
    else:
        inp_img = downsize(img, 256)

    return inp_img

def infer_new_image(input_image):
    cv2.imwrite("temp.jpg", infer_img(input_image))
    print('Resized Dimensions : ',inp_img.shape)
    output_image = life_of_photo("./temp.jpg")
    save_path = "generatedImages/" + input_image.split("/")[-1] + "_GAN_generated.jpg"
    output_image.save(save_path)
    
    resize_img = cv2.imread(save_path)
    generated_image_resized = cv2.resize(resize_img, (test_img_shape[1],test_img_shape[0]))

    output_path = "resizedOutput/" + input_image.split("/")[-1] + "_GAN_Resized.jpg"
    cv2.imwrite(output_path, generated_image_resized)
    
