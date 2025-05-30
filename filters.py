import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter

def bandpass_filter(data, sr, lowcut, highcut, order=6):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def clean_voice(audio, sr, lowcut=300, highcut=3400):
    filtered = bandpass_filter(audio, sr, lowcut, highcut)
    reduced = nr.reduce_noise(y=filtered, sr=sr, prop_decrease=1.0)
    max_val = np.max(np.abs(reduced))
    if max_val > 0:
        reduced = reduced / max_val * 0.95
    return reduced
