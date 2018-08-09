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

# Plotting modules
import matplotlib.pyplot as plt
import matplotlib.style as ms
import matplotlib.ticker as ticker
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.gridspec as gridspec

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
np.random.seed(7)


def mfcc(y, sr=8000, n_mfcc=12):
    '''
    Finds the MFCC coefficients given sampled audio data

    Arguments:
    y - sampled audio signal
    sr - sampling rate (Hz)
    n_mfcc - Number of MFCC coefficients

    Returns:
    list of MFCC coefficients

    '''
    return librosa.feature.mfcc(y=y,sr=sr, n_mfcc=n_mfcc)


def extract_mfccs(y):
    '''
    Extract MFCC coefficients from short duration audio clips

    Arguments:
    y - sampled audio signal

    Returns:
    list of MFCC coefficients for each sub-sample

    '''
    mfccs_list = []
    ran = len(y)//160
    for i in range(ran-10):
        y_clip = y[160*i:160*(i+1)]
        mfccs_clip = mfcc(y_clip)
        mfccs_clip = np.array(mfccs_clip)
        mfccs_clip = mfccs_clip.flatten()
        mfccs_list.append(mfccs_clip)
    return mfccs_list


def read_files(path):
	X = []
	for fn in tqdm(path):
	    y, sr = librosa.load(fn, sr=8000)
	    features = extract_mfccs(y)
	    X.extend(features)
	return X

# Read all the files from the dataset folder

def get_data(path_em='../Data/balanced/cleaned_emergency/', path_nonem='../Data/balanced/nonEmergency/'):
	# # Make the data file path user defined
	# path_em = '../data/balanced/cleaned_emergency/'
	# path_nonem = '../data/balanced/nonEmergency/'

	em_files = glob.glob(os.path.join(path_em, '*.wav'))
	nonem_files = glob.glob(os.path.join(path_nonem, '*.wav'))

	print("Reading the Em class files")
	X_em = read_files(em_files)

	N_em = len(em_files) # Remove data imbalance
	print("Reading the Non-Em class files")
	X_nonem = read_files(nonem_files[:N_em])
	
	return X_em, X_nonem


def prepare_data_train(X_em, X_nonem):
    X_em = np.array(X_em)
    X_nonem = np.array(X_nonem)

    X = np.vstack((X_em, X_nonem))
    Y = np.hstack((np.ones(len(X_em)), np.zeros(len(X_nonem))))

    scaler = StandardScaler()
    scaler.fit_transform(X)
    
    X, Y = shuffle(X, Y, random_state=7)

    # Save scaler for testing later
    # Use sklearn's inbuilt saving tool
    scaler_filename = "scaler.save"
    joblib.dump(scaler, scaler_filename)
    print("Saved scaler!")
    
    return X, Y, scaler


def prepare_data_test(X_em, X_nonem, scaler):
    X_em = np.array(X_em)
    X_nonem = np.array(X_nonem)

    X = np.vstack((X_em, X_nonem))
    Y = np.hstack((np.ones(len(X_em)), np.zeros(len(X_nonem))))

    # # Correct the expression below
    # X = (X - scaler.mean_)/scaler.std_
    scaler.fit_transform(X)

    X, Y = shuffle(X, Y, random_state=7)

    return X, Y


def build_model():
	model = Sequential([
		Dense(24, input_dim=12, activation='relu'),
		Dense(1, activation='sigmoid'),
	])

	model.summary()

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
	callbacks_list = [earlystop]

	return model


def run_model(model, X_train, Y_train, X_test, Y_test):
	history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), batch_size=512)
	model.save("model2.h5")
	print("Saved model to disk!")
	return history


def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best');

    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best');

    plt.savefig('model_history.png')


def clip_level_prediction(model, X_test, Y_test):
	Y_pred = model.predict_classes(X_test)
	cm = confusion_matrix(Y_pred, Y_test)
	df_cm = pd.DataFrame(cm, index = ['Non-EM', 'EM'],
	                  columns = ['Non-EM', 'EM'])
	plt.figure(figsize = (8,6))
	sns.heatmap(df_cm, annot=True, cmap='YlGnBu');
    

def predict_probability(y, scaler):
    mfccs_list = extract_mfccs(y)
    scaler.transform(mfccs_list)
    count = 0
    N = 20 # Window size
    th = 0.5 # Minimum probabilty value for Em presence

    model = load_model('model2.h5')

    prob_list = []
    class_list = []

    for i in range(N):
        p = model.predict(mfccs_list[i].reshape(1,12), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
    prob = np.mean(prob_list)


    if prob > th:
        #print("Em")
        class_list.append(1)
    else:
        #print("Non-em")
        class_list.append(0)

    for i in range(N,len(mfccs_list)):
        prob_list.pop(0)
        p = model.predict(mfccs_list[i].reshape(1,12), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
        prob = np.mean(prob_list)
        #print(prob)
        if prob > th:
            #print("Em")
            class_list.append(1)
        else:
            #print("Non-em")
            class_list.append(0)

    return class_list


# Test Accuracy
def predict_output(y, scaler):
    mfccs_list = extract_mfccs(y)
    scaler.transform(mfccs_list)
    count = 0
    N = 20
    th = 0.5

    model = load_model('model2.h5')

    prob_list = []
    class_list = []
    for i in range(N):
        p = model.predict(mfccs_list[i].reshape(1,12), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
    prob = np.mean(prob_list)
    #print(prob)
    if prob > th:
        #print("Em")
        class_list.append(1)
    else:
        #print("Non-em")
        class_list.append(0)

    for i in range(N,len(mfccs_list)):
        prob_list.pop(0)
        p = model.predict(mfccs_list[i].reshape(1,12), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
        prob = np.mean(prob_list)
        #print(prob)
        if prob > th:
            #print("Em")
            class_list.append(1)
        else:
            #print("Non-em")
            class_list.append(0)
    if np.mean(class_list) > 0.5:
        return 1
    else:
        return 0


def main():

    ## TO DO:
    # Save extracted train and test data into npz or hdfs format

    # train data
    train_path_em = '../Data/balanced/cleaned_emergency/'
    train_path_nonem = '../Data/balanced/nonEmergency/'
    # train_path_em = '../Cleaned data/Train_balanced/Emergency/'
    # train_path_nonem = '../Cleaned_data/Train_balanced/nonEmergency/'

    print("Training data")
    Em_data, Nonem_data = get_data(train_path_em, train_path_nonem)

    X_train, Y_train, scaler = prepare_data_train(Em_data, Nonem_data)

    # test data
    test_path_em = '../Data/new_eval/cleaned_emergency/'
    test_path_nonem = '../Data/eval/nonEmergency/'
    # test_path_em = '../Cleaned data/Eval_balanced/Emergency/'
    # test_path_nonem = '../Cleaned data/Eval_balanced/nonEmergency/'

    print("Test data")
    Em_data, Nonem_data = get_data(test_path_em, test_path_nonem)

    X_test, Y_test = prepare_data_test(Em_data, Nonem_data, scaler)

    # Build the model
    model = build_model()

    # Run the training
    history = run_model(model, X_train, Y_train, X_test, Y_test)

    # Plot the loss and accuracy curves
    plot_model_history(history)

    # Get audio clip level evaluation results
    clip_level_prediction(model, X_test, Y_test)
    
    test_em_files = glob.glob(os.path.join(test_path_em, '*.wav'))
    test_nonem_files = glob.glob(os.path.join(test_path_nonem, '*.wav'))

    tot_em, correct_em, tot_nonem, correct_nonem = 0, 0, 0, 0

    print("Evaluating Em class test data")
    for test_file in tqdm(test_em_files):
        y, sr = librosa.load(test_file, sr=8000)
        classes = predict_output(y, scaler)
        if classes == 1:
            correct_em += 1
        tot_em += 1
    
    print("Evaluating NonEm class test data")
    for test_file in tqdm(test_nonem_files):
        y, sr = librosa.load(test_file, sr=8000)
        classes = predict_output(y, scaler)
        if classes == 0:
            correct_nonem += 1
        tot_nonem += 1

    print("Correct Em: {}\nTotal Em: {}\nCorrect NonEm: {}\nTotal NonEm: {}".format(correct_em, tot_em, correct_nonem, tot_nonem))

    print("=== EVALUTION METRICS ===")
    print("Accuracy: {}".format(float(correct_em + correct_nonem)/(tot_em + tot_nonem)))
    print("Precision: {}".format(float(correct_em)/(tot_nonem - correct_nonem + correct_em)))
    print("Recall: {}".format(float(correct_em)/tot_em))

if __name__ == "__main__":
    main()