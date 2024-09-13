import numpy as np
import librosa
import hparams
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchinfo import summary
from librosa.filters import mel as librosa_mel_fn
import glob
import os
import matplotlib
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
min_level_db= -100

def load_audio(filepath, sr):
    y, sr = librosa.load(filepath, sr=sr)

    return y, sr

def calc_sr_by_UPR(input_sr, UPR):
    target_sr = int(np.ceil(input_sr / UPR))
    return target_sr, int(target_sr/2)

def preprocess_path(path):
    parts = path.split("/")
    if len(parts) > 1:
        new_path = '/'.join(parts[1:])
    else:
        new_path = ''

    return str(new_path)[:-1]

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C
def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def wav_to_log_mel(y, sr, n_mels, normalize):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=24000, power = 2)
    if normalize:
        log_mel = normalize_mel(librosa.power_to_db(S, ref = np.max))
    else:
        log_mel = librosa.power_to_db(S, ref = np.max)

    return log_mel


def mel_spectrogram(y, n_fft, n_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # input = np.array of shape [Batch_size, sequence_length]
    try:
        audio_np = y.cpu().detach().numpy().astype(float)
    except:
        audio_np = y
    mel_spectrogram = librosa.feature.melspectrogram(y = audio_np, sr = sampling_rate, n_fft = n_fft, n_mels = n_mels, fmin = fmin, fmax = fmax, center = False)
    return mel_spectrogram

def normalize_mel(S):
    return np.clip((S-min_level_db)/-min_level_db, 0,1)


def summarize_model(model, input_shape, is_cuda = False, is_bi = False):
    if not is_bi:
        if is_cuda:
            x = torch.rand(input_shape).to("cuda:0")
            flop_counter = FlopCountAnalysis(model, x)
        else:
            x = torch.rand(input_shape)
            flop_counter = FlopCountAnalysis(model, x)
    if is_bi:
        if is_cuda:
            x = torch.rand(input_shape[0]).to("cuda:0")
            x_hat = torch.rand(input_shape[1]).to("cuda:0")
        else:
            x = torch.rand(input_shape[0])
            x_hat = torch.rand(input_shape[1])
    print(parameter_count_table(model))
    summary(model, input_size=input_shape)


def calc_receptive_field(kernel_sizes, stride_sizes):

    if len(kernel_sizes) != len(stride_sizes):
        raise ValueError("The length of kernel_sizes and stride_sizes must be the same")

    receptive_field = 0
    for l in range(len(kernel_sizes)):
        product = 1
        for i in range(l):
            product *= stride_sizes[i]
        receptive_field += (kernel_sizes[l] - 1) * product

    return receptive_field + 1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def trim_or_pad(arr, k):

    sequence_length = arr.shape[0]
    remainder = sequence_length % k

    if remainder == 0:
        # Already divisible by k
        return arr
    elif sequence_length > k:
        # Trim the array to be divisible by k
        trimmed_length = sequence_length - remainder
        return arr[:trimmed_length]
    else:
        # Pad the array with zeros to be divisible by k
        padding_length = k - remainder
        padded_arr = np.pad(arr, (0, padding_length), mode='constant', constant_values=0)
        return padded_arr



