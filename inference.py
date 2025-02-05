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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"
    print("setup initialized...")
def init_process(rank):
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://localhost:12345',
                                         world_size=4,
                                         rank=rank, timeout=timedelta(minutes=30))

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

with open("cfgs.json", "r") as file:
    json_config = json.load(file)
    cfg = AttrDict(json_config)
ckpt_waveumamba = "your_checkpoint_path"

val_names = "val-files.txt"
output_dir = cfg.data["etc_save_path"]
device = torch.device("cuda")
#checkpoint = prefix_load_checkpoint(ckpt_path, device)

#missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

#state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

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

def inference(a, count = 5, type = 1):
    global device
    setup(0, 8)
    wandb.init(project="project_name")
    wandb.require("core")
    random.seed(cfg.info["seed"])
    #global cfg
    print("wandb initialized...")
    generator = Generator(cfg.mamba).to(device)
    print("model initialized...")
    mother_dir = cfg.data["dir"]
    val_data_files = cfg.data["val_names"]
    UPR = cfg.audio["UPR"]
    state_dict_g = prefix_load_checkpoint(a.checkpoint_file, {'cuda:%d' % 0: 'cuda:%d' % 1})
    generator.load_state_dict(state_dict_g, strict= True)
    lsd_list = []
    time_list = []
    with open(val_data_files, "r") as f:
        files = f.readlines()

    os.makedirs(a.output_dir, exist_ok=True)
    files = [os.path.join(mother_dir, utils.preprocess_path(filepath)) for filepath in files if
             os.path.exists(os.path.join(mother_dir, utils.preprocess_path(filepath)))]
    generator.eval()
    generator.remove_weight_norm()

    files = random.sample(files, count)


    with torch.no_grad():
        for i, filename in tqdm(enumerate(files)):

            y_high, sr_high = utils.load_audio(filename, sr = 48000)
            y_high = utils.trim_or_pad(y_high, 128)
            #y_high = normalize(y_high) * 0.9
            gt_high_mel = librosa.power_to_db(
                utils.mel_spectrogram(y_high, cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                cfg.audio["hop"],
                                cfg.audio["window_size"], cfg.audio["fmin"],
                                cfg.audio["fmax_for_loss"]), ref=np.max)
            upsampling_ratio = random.choice(UPR)
            sr_low_calc, cutoff_freq = utils.calc_sr_by_UPR(sr_high, upsampling_ratio)
            gt_high_audio = wandb.Audio(y_high, sample_rate=cfg.audio["sr"], caption="ground_truth_high_audio")

            if type == 1:
                y_rebuilt, sr_low = filter.build_lowres(y_high, sr_high, cutoff_freq)
                y_low, sr_low = filter.resample_audio(y_rebuilt, sr_high, sr_low_calc)
                y_rebuilt, sr_low = filter.resample_audio(y_low, sr_low, sr_high)
                if y_rebuilt.shape[0] > y_high.shape[0]:
                    y_rebuilt = y_rebuilt[:-(y_rebuilt.shape[0] - y_high.shape[0])]
                elif y_rebuilt.shape[0] < y_high.shape[0]:
                    y_rebuilt = y_rebuilt + [0] * (y_high.shape[0] - y_rebuilt.shape[0])
            elif type == 2:
                y_low, sr_low = filter.resample_audio(y_high, sr_high, sr_low_calc)
                y_rebuilt, sr_low = filter.resample_audio(y_low, sr_low, sr_high)
                if y_rebuilt.shape[0] > y_high.shape[0]:
                    y_rebuilt = y_rebuilt[:-(y_rebuilt.shape[0] - y_high.shape[0])]
                elif y_rebuilt.shape[0] < y_high.shape[0]:
                    y_rebuilt = y_rebuilt + [0] * (y_high.shape[0] - y_rebuilt.shape[0])

            else:

                y_rebuilt, sr_low = filter.build_lowres(y_high, sr_high, cutoff_freq)

            gt_low_mel = librosa.power_to_db(
                utils.mel_spectrogram(y_rebuilt, cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                      cfg.audio["hop"],
                                      cfg.audio["window_size"], cfg.audio["fmin"],
                                      24000), ref=np.max)
            gt_low_audio = wandb.Audio(y_rebuilt, sample_rate=cfg.audio["sr"], caption="ground_truth_low_audio")
            y_low = torch.FloatTensor(y_rebuilt).to(device)

            start_time = time.time()
            y_g_hat = generator(y_low.unsqueeze(0).unsqueeze(0))
            end_time = time.time()
            audio = y_g_hat.squeeze()
            audio = audio.cpu().numpy()

            # Log Spectral Distance
            gt_stft = wav_to_spectrogram(y_high, cfg.audio["hop"], cfg.audio["nfft"])
            pred_stft = wav_to_spectrogram(audio, cfg.audio["hop"], cfg.audio["nfft"])
            LSD = lsd(pred_stft, gt_stft)
            lsd_list.append(LSD.item())

            inf_time = end_time - start_time
            time_list.append(inf_time)
            pred_mel = librosa.power_to_db(
                utils.mel_spectrogram(audio, cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                cfg.audio["hop"],
                                cfg.audio["window_size"], cfg.audio["fmin"],
                                24000), ref=np.max)
            pred_audio = wandb.Audio(audio, sample_rate=cfg.audio["sr"], caption="predicted_audio")
            gt_high_mel = wandb.Image(
                librosa.display.specshow(gt_high_mel, sr=cfg.audio["sr"], x_axis='time', y_axis='mel',
                                         fmax=24000), caption="gt_high_mel")
            gt_low_mel = wandb.Image(librosa.display.specshow(gt_low_mel, sr=cfg.audio["sr"], x_axis='time',
                                                              y_axis='mel', fmax=24000),
                                     caption="gt_low_mel")
            pred_mel = wandb.Image(librosa.display.specshow(pred_mel, sr=cfg.audio["sr"], x_axis='time',
                                                            y_axis='mel', fmax=24000),
                                   caption="predicted_mel")

            wandb.log({f"ground truth of {filename[-12:]} th high mel": gt_high_mel})
            wandb.log({f"ground truth of {filename[-12:]} th low mel": gt_low_mel})
            wandb.log({f"predicted {filename[-12:]} th mel": pred_mel})
            wandb.log({f"ground truth of {filename[-12:]} th high audio": gt_high_audio})
            wandb.log({f"ground truth of {filename[-12:]} th low audio": gt_low_audio})
            wandb.log({f"predicted {filename[-12:]} th audio , UPR: {upsampling_ratio}, LSD :{LSD.item()}": pred_audio})
            output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated_e2e.wav')
            write(output_file, cfg.audio["sr"], audio)


            print(output_file)
            print("inference finished...")
        mean_lsd = np.mean(np.array(lsd_list))
        mean_time = np.mean(np.array(time_list))
        print(f"mean LSD is : {mean_lsd}, mean time is : {mean_time}")





def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_file', default=val_names)
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', default = 'your_checkpoint_path')
    a = parser.parse_args()

    with open("cfgs.json", "r") as file:

        json_config = json.load(file)
        h = AttrDict(json_config)
        print("config initialized..")
    torch.manual_seed(h.info["seed"])
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.info["seed"])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    inference(a)


if __name__ == '__main__':
    main()
