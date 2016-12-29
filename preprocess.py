import numpy as np
from scipy.misc import imresize

#returns preprocessed value of most recent frame
def preprocess(seq):
	if len(seq) <= 1:
		#get most recent frame
		frame = seq[len(seq) - 1]
		#get previous frame
		frame_prev = frame
	else:
		#get most recent frame
		frame = seq[len(seq) - 1]
		#get previous frame
		frame_prev = seq[len(seq) - 3]
	return get_resize_from_lum(get_luminescence(get_max_value(frame, frame_prev)))

def get_max_value(frame, frame_prev):
	return np.maximum(frame, frame_prev)

def get_luminescence(frame):
	R = frame[:,:, 0]
	G = frame[:, :, 1]
	B = frame[:, :, 2]
	return 0.2126*R + 0.7152*G + 0.0722*B

def get_resize_from_lum(lum_frame):
	return imresize(lum_frame, (84, 84))

