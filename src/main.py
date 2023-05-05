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

def create_mfcc_file_for_test(filename):
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
    #Check all files in ressources folder and is different from .mfcc
    allFiles = [f for f in listdir(path) if isfile(join(path, f)) and f[-5:] != ".mfcc"]

    return allFiles
def getFileList(path:str):

    #Get all oui_... files in resssources folder in an array
    allFiles = []
    #Check all files in ressources folder
    for i in range(1,NUMBER_OF_FILE+1):
        file = path + str(i) + ".wav"
        allFiles.append(file)
    return allFiles

def createRefFolder(fileList, fileName, isAudio):
    file = open(fileName, "w")
    for i in range(len(fileList)):
        if isAudio:
            file.write("ressources/audio/" + fileList[i] + "\n")
        else:
            file.write(fileList[i] + "\n")
    file.close()

#Pour la distance locale d, nous prendrons la distance Euclidienne au carré Σ(X|i][k] – Y[j][k])² / (ΣX[i][k]² . ΣY[i][k]²)
#
def local_distance(X, Y):
    if len(X) != len(Y):
        raise ValueError("Les deux vecteurs doivent avoir la même taille")

    diff_square_sum = 0
    X_square_sum = 0
    Y_square_sum = 0

    for i in range(len(X)):
        diff_square_sum += (X[i] - Y[i]) ** 2
        X_square_sum += X[i] ** 2
        Y_square_sum += Y[i] ** 2

    if X_square_sum == 0 or Y_square_sum == 0:
        return 0

    result = diff_square_sum / (X_square_sum * Y_square_sum)
    return result

def dtw(ref, test):
    n, m = len(ref), len(test)
    dtw_matrix = np.zeros((n + 1, m + 1))

    for i in range(1, n + 1):
        dtw_matrix[i, 0] = 0
    for j in range(1, m + 1):
        dtw_matrix[0, j] = 0

    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            last_min = np.min([dtw_matrix[i - 1, j],
                               dtw_matrix[i, j - 1],
                               dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = last_min + local_distance(ref[i - 1], test[j - 1])

    return dtw_matrix





def main():


    createRefFolder(getFileList("ressources/oui_0"), "ressources/ref/Ref_OUI.txt", False)
    createRefFolder(getFileList("ressources/non_0"), "ressources/ref/Ref_NON.txt", False)
    createRefFolder(getFileListFromFolder("ressources/audio/"), "ressources/ref/Test.txt", True)


    #Get ref oui
    refOui = []
    file = open("ressources/ref/Ref_OUI.txt", "r")
    for line in file:
        refOui.append(line[:-1])
    file.close()

    #Get ref non
    refNon = []
    file = open("ressources/ref/Ref_NON.txt", "r")
    for line in file:
        refNon.append(line[:-1])
    file.close()

    #Get test
    test = []
    file = open("ressources/ref/Test.txt", "r")
    for line in file:
        test.append(line[:-1])
    file.close()

    for i in range(len(test)):
        create_mfcc_file_for_test(test[i])

    for i in range(len(refOui)):
        create_mfcc_file(refOui[i])
    for i in range(len(refNon)):
        create_mfcc_file(refNon[i])

    #D'abord pour les RefOui
    resultOui = getResultFromFiles(refOui, test[1])
    moyenneOuiBon = []

    for X in range(len(resultOui)):
        moyenneOuiBon.append(np.average(resultOui[X]))

    print("Moyenne Oui Bon : " + str(np.average(moyenneOuiBon)))


    resultNon = getResultFromFiles(refNon, test[1])
    moyenneNonBon = []
    for X in range(len(resultNon)):
        moyenneNonBon.append(np.average(resultNon[X]))

    print("Moyenne Non Bon : " + str(np.average(moyenneNonBon)))




def getResultFromFiles(refFile, testFile):
    result = []
    clearedTestFile = clearMfcc(testFile)
    for i in range(len(refFile)): #Pour chaque fichier de ref
        clearedRefFile = clearMfcc(refFile[i])
        dtwvar = dtw(clearedRefFile, clearedTestFile)
        result.append(dtwvar)
    return result
def clearMfcc(filename):
    file = open(filename[:-4] + ".mfcc", "r")
    # Remove first line and all "Vecteur N : "
    lines = file.readlines()[1:]
    cleaned_vectors = []

    # Parcourir chaque ligne et enlever "Vecteur nb :"
    for line in lines:
        # Trouver l'index de ':' et extraire la partie après cet index
        colon_index = line.find(':')
        cleaned_line = line[colon_index + 1:].strip()

        # Convertir la chaîne en liste de nombres
        vector = [float(num) for num in cleaned_line.split()]

        # Ajouter le vecteur nettoyé à la liste
        cleaned_vectors.append(vector)

    return cleaned_vectors






if __name__ == "__main__":
    main()


