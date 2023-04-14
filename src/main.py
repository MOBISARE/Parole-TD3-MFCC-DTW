import librosa
import os.path
from typing import Final, Tuple, List

import numpy as np

NON_01: Final[str] = os.path.join("ressources", "non_01.wav")


def hamming(size):
    return np.hamming(size)


def preaccentuation(signal):
    signaldec = np.roll(signal, 1)  # décale les éléments vers +1 (le dernier passe à 0)
    signaldec[0] = 0
    output = signal - 0.97 * signaldec
    return output


def run():
    mon_signal, frequence_signal = librosa.load(NON_01)
    librosa.display.waveshow(mon_signal, sr=frequence_signal)
    window_length = 25 * frequence_signal // 1000
    hop_length = 10 * frequence_signal // 1000
    fen = hamming(window_length)
    print(window_length)
    print(hop_length)
    mon_signal_preaccentue = preaccentuation(mon_signal)
    mfccs = librosa.feature.mfcc(mon_signal_preaccentue, sr=frequence_signal, win_length=window_length, hop_length=hop_length, n_mfcc=13, window=fen)
    mfcc_dela = librosa.feature.delta(mfccs)
    mffcs_delta_2 = librosa.feature.delta(mfccs, order=2)


if __name__ == '__main__':
    run()
