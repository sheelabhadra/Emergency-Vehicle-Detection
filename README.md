# Emergency-Vehicle-Detection
This repository contains implementation of papers on sound-based Emergency Vehicle Detection in Python.  

## Paper-1
[A real-time siren detector to improve safety of guide in traffic environment](https://ieeexplore.ieee.org/document/7080691/)

This paper used only 1 feature i.e. the periodicity of the audio signal to identify the presence of a siren. Emergency signals in general have a periodic pattern which repeats over a short duration of time. Additionally, they also belong to specific frequency bands which lends them the high pitch sound that they are usually associated with. The paper exploited these characteristics pertaining to emergency signals by using a pitch detection algorithm. The pitch detection algorithm is a time domain technique which aims to categorize each portion of an audio signal as pitched or unpitched. A Modified Pitch Detection (MDF) algorithm which is a less computationally expensive version of the autocorrelation function is used to find the pitch. Using the MDF vs the lag plot, the pitch of the signal is detected. A pitch vs time plot is then used to determine the pitch found in each small audio sample of size 512 samples. An emergency signal estimator that calculates the fraction of time for which the pitch remains inside the frequency band for emergency signals predicts the presence of an emergency signal in the audio sample.

There are better versions of the MDF algorithm which make the pitch detection more accurate. CAMDF (Circular Average Magnitude Difference Function) was used to calculate the pitch in our case as it resulted in more accurate results.  

It was observed that the algorithm performed well only if the profile of the emergency signals is known, stay fairly constant, and do not have a lot of ambient noise added. But in real life scenarios this is seldom the case. On the data that we used to test the algorithm, there were a lot of false negatives and false positives as many of the audio samples had a wide variety of emergency sounds. At times the noise in the audio samples made it hard for the algorithm to identify emergency signals.  

The pitch detection technique is a simple approach that assumes that all emergency signals are periodic in nature. But, this is not the case in real-world situations as many emergency signals do not follow this pattern. Moreover, a lot of times emergency signals are mixed with other sounds and noise that makes it difficult for the algorithm to identify the pitch of the emergency signal.  

One way to improve the results would be to look at features that are more representative of emergency signals and also robust to some level of noise. The range of frequency at which emergency signals operate is an important factor that could be considered. Using a simple bandpass filter might be helpful in eliminating some of the noise and make it easier to focus only on the frequency range where emergency sounds lie. Mel Frequency Cepstral Coefficients (MFCCs) are known to perform very well on speech recognition tasks. They are a good representation of the audio signals and follow some of the basic principles in which the human auditory system works. But, since these features can’t be interpreted by just looking at their values, they are fed to a machine learning classifier. The classifier learns to detect patterns in the input data to distinguish between emergency signals and non-emergency signals. One such approach has been used in the paper described in the next section.


## Paper-2
[An Automatic Emergency Signal Recognition System for the Hearing Impaired](https://ieeexplore.ieee.org/document/4041054/)

## Paper-3
[Detection of alarm sounds in noisy environments](https://ieeexplore.ieee.org/abstract/document/8081527/)


