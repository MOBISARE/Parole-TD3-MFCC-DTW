from os import listdir
from os.path import isfile, join

import numpy as np
from scipy.io.wavfile import read, write
#Import librosa to use librosa.load
import librosa
import matplotlib.pyplot as plt
from typing import Final, Tuple, List

N_MFCC: Final[int] = 13
NUMBER_OF_FILE: Final[int] = 8
def preaccentuation(signal):
    signaldec = np.roll(signal,1)
    signaldec[0] = 0
    output = signal - 0.97*signaldec
    return output
def fenetrageHamming(size):
    return np.hamming(size)

def create_mfcc_file(filename):
    x, fs = librosa.load(filename)

    win_length = 25 * fs // 1000  # = 25 ms = length of a time frame
    hop_length = 10 * fs // 1000  # = 10 ms = frame periodicity
    windows = fenetrageHamming(win_length)  # Fenêtre de Hamming
    y = preaccentuation(x)
    mfccs = librosa.feature.mfcc(y=y, sr=fs, win_length=win_length, hop_length=hop_length, n_mfcc=N_MFCC,
                                 window=windows)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    # Remove the first row of mfccs and mfccs_delta and mfccs_delta2 because its the energy
    mfccs = np.delete(mfccs, 0, 0)
    mfccs_delta = np.delete(mfccs_delta, 0, 0)
    mfccs_delta2 = np.delete(mfccs_delta2, 0, 0)

    # Create a string that will store the mfccs

    mfccs = mfccs.T
    mfccs_delta = mfccs_delta.T
    mfccs_delta2 = mfccs_delta2.T
    mfccs_string = str(len(mfccs)) + "\n"
    for i in range(len(mfccs)):
        mfccs_string += "Vecteur " + str(i + 1) + " : "
        for j in range(len(mfccs[i])):
            mfccs_string += str(mfccs[i][j]) + " "
        for z in range(len(mfccs_delta[i])):
            mfccs_string += str(mfccs_delta[i][z]) + " "
        for k in range(len(mfccs_delta2[i])):
            mfccs_string += str(mfccs_delta2[i][k]) + " "
        mfccs_string += "\n"

    # remove .wav from file name into a new variable
    filename = filename[:-4]
    # create a new file with the same name as the audio file but with .mfcc extension
    file = open(filename + ".mfcc", "w")
    file.write(mfccs_string)
    file.close()


def getFileListFromFolder(path:str):

    #Get all files from folder ressources/audio in array
    allFiles = []
    #Check all files in ressources folder
    allFiles = [f for f in listdir(path) if isfile(join(path, f))]

    return allFiles
def getFileList(path:str):

    #Get all oui_... files in resssources folder in an array
    allFiles = []
    #Check all files in ressources folder
    for i in range(1,NUMBER_OF_FILE+1):
        file = path + str(i) + ".wav"
        allFiles.append(file)
    return allFiles

def createRefFolder(fileList, fileName):
    print(fileList)
    print(fileName)
    file = open(fileName, "w")
    for i in range(len(fileList)):
        file.write(fileList[i] + "\n")
    file.close()

#Pour la distance locale d, nous prendrons la distance Euclidienne au carré
#Σ(X|i][k] – Y[j][k])² / (ΣX[i][k]² . ΣY[i][k]²)
def d(X,Y):
    return np.sum((X-Y)**2)/(np.sum(X**2)*np.sum(Y**2))
def dtw(ref, test):
    n, m = len(test), len(ref)
    nb = N_MFCC

    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s[i - 1] - t[j - 1])

            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix




def main():

    for i in range(1,NUMBER_OF_FILE+1):
        fileNon = "ressources/non_0" + str(i) + ".wav"
        fileOui = "ressources/oui_0" + str(i) + ".wav"
        create_mfcc_file(fileNon)
        create_mfcc_file(fileOui)

    createRefFolder(getFileList("ressources/oui_0"), "ressources/ref/Ref_OUI.txt")
    createRefFolder(getFileList("ressources/non_0"), "ressources/ref/Ref_NON.txt")
    createRefFolder(getFileListFromFolder("ressources/audio/"), "ressources/ref/Test.txt")


if __name__ == "__main__":
    main()


