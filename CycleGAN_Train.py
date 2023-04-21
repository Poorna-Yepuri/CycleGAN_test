"""
This is for processing images and calling the CycleGAN model definition for training.

"""

from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import resample

# load all images in a directory into memory
def load_images(path, size=(512,512)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)


# dataset path
path = 'HighRes_Data/'


# load dataset A - Monet paintings
dataA_all = load_images(path + 'DomainA/')
print('Loaded dataA_all: ', dataA_all.shape)

from sklearn.utils import resample
# To get a subset of all images, for faster training during demonstration
dataA = resample(dataA_all, 
                 replace=False,     
                 n_samples=1000,    
                 random_state=8)
print('Loaded dataA: ', dataA.shape)

# load dataset B - Photos 
dataB_all = load_images(path + 'DomainB/')
print('Loaded DomainB: ', dataB_all.shape)

dataB = resample(dataB_all, 
                 replace=False,     
                 n_samples=1000,    
                 random_state=8)
print('Loaded dataB: ', dataB.shape)


# plot source images
n_samples = 3
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + i)
	plt.axis('off')
	plt.imshow(dataA[i+12].astype('uint8'))
# plot target image
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + n_samples + i)
	plt.axis('off')
	plt.imshow(dataB[i+12].astype('uint8'))
plt.show()



# load image data
data = [dataA, dataB]

print('Loaded', data[0].shape, data[1].shape)

#Preprocess data to change input range to values between -1 and 1
# This is because the generator uses tanh activation in the output layer
#And tanh ranges between -1 and 1
def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,512] to [-1,1]
	X1 = (X1 - 256) / 256
	X2 = (X2 - 256) / 256
	return [X1, X2]

dataset = preprocess_data(data)

from Cycle_GAN_Model import define_generator, define_discriminator, define_composite_model, train
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

from datetime import datetime 
start1 = datetime.now() 
# train models
# loss_df = pd.DataFrame(columns=['iteration', 'dA_loss1', 'dA_loss2', 'dB_loss1', 'dB_loss2', 'g_loss1', 'g_loss2'])
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=10)

stop1 = datetime.now()
#Execution time of the model 
execution_time = stop1-start1
print("Execution time is: ", execution_time)

