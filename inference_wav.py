from __future__ import absolute_import, division, print_function, unicode_literals
import utils
import glob
import os
import numpy as np
import wandb
import filter
import random
import argparse
import librosa
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from modules.waveumamba import waveumamba as Generator
from datetime import timedelta
from tqdm import tqdm
import time
h = None
EPS = 1e-12


def prefix_load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint = torch.load(filepath, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    prefix = "module."
    for key in state_dict["generator"].keys():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
            new_state_dict[new_key] = state_dict["generator"][key]

    print("Complete.")
    return new_state_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def wav_to_spectrogram(wav, hop_length, n_fft):
    f = np.abs(librosa.stft(wav, hop_length=hop_length, n_fft=n_fft, center = False))
    f = np.transpose(f, (1, 0))
    f = torch.tensor(f[None, None, ...])
    return f


def lsd(est ,target):
    assert est.shape == target.shape, "Spectrograms must have the same shape."
    est = est.squeeze(0).squeeze(0) ** 2
    target = target.squeeze(0).squeeze(0) ** 2
    # Compute the log of the magnitude spectrograms (adding a small epsilon to avoid log(0))
    epsilon = 1e-10
    log_spectrogram1 = torch.log10(target + epsilon)
    log_spectrogram2 = torch.log10(est + epsilon)
    squared_diff = (log_spectrogram1 - log_spectrogram2) ** 2
    squared_diff = torch.mean(squared_diff, dim = 1) ** 0.5
    lsd = torch.mean(squared_diff, dim = 0)
    return lsd

def infer(a, cfg, type = 1):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.info["seed"])
        print("cuda initialized...")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print("configurations initialized...")
    random.seed(cfg.info["seed"])
    generator = Generator(cfg.mamba).to(device)
    print("model initialized...")
    state_dict_g = prefix_load_checkpoint(a.checkpoint_file, {'cuda:%d' % 0: 'cuda:%d' % 1})
    print("checkpoint successfully loaded...")
    generator.load_state_dict(state_dict_g, strict= True)
    lsd_list = []
    time_list = []


    with torch.no_grad():
        
        prefix = a.wav_path[-7:-3]
        y_low, sr_high = utils.load_audio(a.wav_path, sr = 48000)
        y_low = utils.trim_or_pad(y_low, 128).to(device)
           



        start_time = time.time()
        y_g_hat = generator(y_low.unsqueeze(0).unsqueeze(0))
        end_time = time.time()
        audio = y_g_hat.squeeze()
        audio = audio.cpu().numpy()

        inf_time = end_time - start_time
        print(f"Inference took {inf_time} seconds")

        output_file = os.path.join(a.output_dir, prefix + '_superresolved.wav')
        write(output_file, cfg.audio["sr"], audio)


        print(output_file)
        print("inference finished...")





def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path', default=None)
    parser.add_argument('--output_dir', default='/outputdir')
    parser.add_argument('--checkpoint_file', default = '/generator.pt')
    parser.add_argument('--cfgs_path', default='/ckpts.json')
    a = parser.parse_args()

    with open(a.cfgs_path, "r") as file:

        json_config = json.load(file)
        h = AttrDict(json_config)
        print("config initialized..")
    torch.manual_seed(h.info["seed"])
    
    infer(a, h)


if __name__ == '__main__':
    main()
