# Import the libraries

# Python's in-built modules
import os
import glob
import time

# External modules
import numpy as np
import pandas as pd
from tqdm import tqdm

# Librosa for audio
import librosa
import librosa.display

# For designing the band-pass filter
from scipy.signal import butter, lfilter, hilbert

# Plotting modules
import matplotlib.pyplot as plt
import matplotlib.style as ms
import matplotlib.ticker as ticker
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.gridspec as gridspec

# Pyaudionalaysis functions
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

# Scikit-learn modules
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

# Keras and TensorFlow modules
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th')

# Supress Tensorflow error logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# fix random seed for reproducibility
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# Bandpass filter to remove noise
def butter_bandpass_filter(data, lowcut=500, highcut=1500, fs=8000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y




