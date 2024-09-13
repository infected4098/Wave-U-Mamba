from scipy.signal import butter, lfilter, cheby1, resample
import numpy as np
import librosa


def butter_bandpass(x, lowcut, highcut, input_sr, order=8):
    nyq = input_sr / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, x)
    return y

def chebyshev_bandpass(x, input_sr, cutoff_freq, order = 8):
  nyq_freq = input_sr / 2
  target_sr = np.round(2*cutoff_freq)
  rp = 0.1 # minor
  wn = cutoff_freq / nyq_freq
  b, a = cheby1(order, rp, wn, btype = "low",
                       analog = False, output = "ba")
  y = lfilter(b, a, x)

  return y, target_sr


def resample_audio(x, input_sr, target_sr):
    # default resampling operation using Librosa
  ratio = float(target_sr) / input_sr
  n_samples = int(np.ceil(x.shape[-1] * ratio))
  y = resample(x, n_samples, axis = -1)
  return y, target_sr


def polyphase_subsample(x, input_sr, target_sr):
  assert input_sr >= target_sr
  y = librosa.resample(x, orig_sr = input_sr, target_sr = target_sr, res_type = "polyphase")
  return y

def build_lowres(x, input_sr, cutoff_freq, order = 8):
  y, target_sr = chebyshev_bandpass(x, input_sr, cutoff_freq, order)

  return y, target_sr




