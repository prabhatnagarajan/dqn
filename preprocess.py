import numpy as np
from scipy.misc import imresize

def get_max_value(frame, frame_prev):
	return np.maximum(frame, frame_prev)

def get_luminance(frame):
	R = frame[:,:, 0]
	G = frame[:, :, 1]
	B = frame[:, :, 2]
	return 0.2126*R + 0.7152*G + 0.0722*B

def get_resize_from_lum(lum_frame):
	return imresize(lum_frame, (84, 84))

