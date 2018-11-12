# Stream audio from microphone
# Classify chunks of audio

###### Importing the libraries #######
from em_detection import *

import os
import numpy as np
import pyaudio
from scipy.signal import butter, lfilter, hilbert

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
# from alarm_train import preprocess, butter_bandpass_filter

from keras.models import load_model
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th')

# Supress Tensorflow error logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

###### Data Pre-processing functions ###

def butter_bandpass_filter(data, lowcut=500, highcut=1500, fs=44100, order=5):
		nyq = 0.5 * fs
		low = lowcut / nyq
		high = highcut / nyq

		b, a = butter(order, [low, high], btype='band')
		y = lfilter(b, a, data)
		return y

def preprocess(y):
		y_filt = butter_bandpass_filter(y)
		analytic_signal = hilbert(y_filt)
		amplitude_envelope = np.abs(analytic_signal)
		return amplitude_envelope

#######################################

RATE = 44100
CHUNK = int(0.1*RATE)
model = load_model('model3_44KHz.h5')

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,  
		 frames_per_buffer=CHUNK)

N = 10
th = 0.5

# Data standardization values
mean = [5.19603840e-02, 2.04577778e-02, 3.12241896e+00, 1.15469393e-01,
	 1.48082321e-01,   5.31328356e-01,   1.24768665e-02,   8.45027338e-02,
	-2.68466708e+01,   3.74163978e+00,  -7.62308678e-01,   8.27776331e-02,
	 1.34093590e-01,   1.71638982e-01,  -4.12313091e-02,   5.10990992e-02,
	 4.01010778e-02,   3.05182617e-02,  -7.21860340e-04,   2.82520453e-02,
	 1.82386609e-03,   9.28361365e-03,   4.06266574e-03,   4.85046986e-02,
	 8.09642983e-03,   1.97478773e-02,   6.24005329e-03,   3.27579647e-02,
	 2.27016929e-03,   4.45466670e-03,  7.55511839e-03,   1.27528544e-01,
	 4.67327577e-03,   4.20225313e-02]

std = [ 0.04192774,  0.02571282,  0.29373228,  0.04122611,  0.03732406,  0.43174152,
	0.03828986,  0.07094834,  4.37741615,  0.88740987,  0.56351395,  0.29495966,
	0.27515089,  0.24544091,  0.2319935,   0.22617261,  0.21444065,  0.20290731,
	0.1863855,  0.15837307, 0.14641837,  0.0145218,   0.00728451,  0.0631253,
	0.01346965,  0.02934718,  0.01045911,  0.04315424,  0.00510846,  0.00800946,
	0.01103524,  0.05733724,  0.00829243,  0.01591553]


prob_list = []
count = 0

scaler_filename = "scaler.save"
scaler = joblib.load(scaler_filename)

try:
	while True:
		count += 1
		
		data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
		y = preprocess(data)
		features_list = audioFeatureExtraction.stFeatureExtraction(y, RATE, CHUNK, CHUNK)
		# print features_list
		# features = (features_list[0] - mean)/std
		# features = scaler.transform(features_list[0].reshape(1,-1))
		p = model.predict(features_list[0].reshape(1,34), batch_size=None, verbose=0)
		p = p.flatten()
		prob_list.append(p)

		if count%10 == 0:
			prob = np.mean(prob_list)

			if prob >= th:
				print("Em - {}".format(prob))
			else:
				print("Non-Em - {}".format(prob))

			prob_list = [prob]

except KeyboardInterrupt:
	stream.stop_stream()  
	stream.close()  
	p.terminate() 
