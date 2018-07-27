from pitch_detection import *

def main():
    # Reading the audio file
    file = '../Cleaned data/Train_balanced/Emergency/17.wav'
    y, sr = librosa.load(file)

    # Run pitch detection
    pitch = Pitch_Detect(y, sr=8000, Ws=512)
    pitch.run()

if __name__ == "__main__":
    main()