import numpy as np
from scipy.misc import imresize
from scipy.misc import imshow
import cv2

'''
Arguments - inputs two grayscale images that we take the maximum of
'''
#TODO REMOVE THIS METHOD
def process(grayscale1, grayscale2):
	return resize(get_max_value(grayscale1, grayscale2))

#returns preprocessed value of most recent frame
#TODO, remove this method
def preprocess(frame1, frame2):
	return resize(grayscale(np.maximum(frame1, frame2)))

#Takes in an rgb image returns the grayscale
def grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def get_luminescence(frame):
	R = frame[:,:, 0]
	G = frame[:, :, 1]
	B = frame[:, :, 2]
	return 0.2126*R + 0.7152*G + 0.0722*B

#def resize(lum_frame):
#	return imresize(lum_frame, (84, 84))

def resize(image):
	return cv2.resize(image, (84, 84))