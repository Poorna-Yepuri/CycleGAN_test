# Required Libraries
import cv2
from os import listdir
from os.path import isfile, join
from pathlib import Path

def downsize(img,size):
    resized_dimensions = (size, size)
    # Create resized image using the calculated dimensions
    resized_image = cv2.resize(img, resized_dimensions,
                                interpolation=cv2.INTER_AREA)

    return resized_image

def upscale(img, img_shape):
    resized_image = cv2.resize(img, (img_shape[1], img_shape[0]),
                                interpolation=cv2.INTER_CUBIC)
    return resized_image
