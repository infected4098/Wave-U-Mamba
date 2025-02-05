# This training code is for a multi-GPU scheme. Please change the code if you want to train the model in a single-GPU environment. 

import warnings
from env import AttrDict, build_env
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import numpy as np
import librosa
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from data.dataset import VCTKMultiSpkDataset
from modules.waveumamba import waveumamba
import auraloss
from utils import scan_checkpoint, load_checkpoint, save_checkpoint, mel_spectrogram, summarize_model
import wandb
from datetime import timedelta
torch.backends.cudnn.benchmark = True
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL" # For debugging

# https://github.com/csteinmetz1/auraloss
loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 8192],
    hop_sizes=[256, 512, 2048],
    win_lengths=[1024, 2048, 8192],
    scale="mel",
    n_bins=128,
    sample_rate=48000,
    perceptual_weighting=True,
)

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim, warmup_steps: int, base_lr: float, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linearly increase the learning rate
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.optimizer.param_groups]
        else:
            # Use the base learning rate after warmup
            return [self.base_lr for _ in self.optimizer.param_groups]


def train(rank, cfg, a):
    print(f"Running DDP on rank {rank}.")
    setup(rank, 4)

    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://127.0.0.1:12345',
                                         world_size=2,
                                         rank=rank, timeout = timedelta(minutes=30))
    #Wandb logging
    wandb.init(project=f"{a.experiment_name}")
    wandb.require("core")
    wandb.run.name = ""
    wandb.config.update(cfg)

    # Early Stopping
    init_err = 99999
    best_err_for_ckpt = 99999
    best_steps = 0
    early_stopping_count = 0
    steps = 0
    warmup_epoch = 5

    #setup
    torch.cuda.manual_seed(cfg.info["seed"])
    device = torch.device('cuda:{:d}'.format(rank))


    #Parallel training
    trainset = VCTKMultiSpkDataset(hparams = cfg, type = 1) #training
    train_sampler = DistributedSampler(dataset = trainset, shuffle=True)
    train_loader = DataLoader(trainset, num_workers=cfg.train["num_workers"], shuffle=False,
                              sampler=train_sampler,
                              batch_size=cfg.train["batch_size"],
                              pin_memory=True,
                              drop_last=True)


    validset = VCTKMultiSpkDataset(hparams = cfg, cv = 1, type = 1) #validation
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

    # Model DDP
    generator = waveumamba(cfg.mamba).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    generator = DistributedDataParallel(module = generator, device_ids = [rank], find_unused_parameters=False).to(device) # find_unused_parameters=True when debugging
    mpd = DistributedDataParallel(module=mpd, device_ids=[rank], find_unused_parameters=False).to(device)
    msd = DistributedDataParallel(module=msd, device_ids=[rank], find_unused_parameters=False).to(device)

    #Checkpoints checking
    cp_g, cp_do = None, None

    # Summarizing models
    if rank == 0:
        print(summarize_model(generator, [2, 1, 25600], is_cuda = True))
        print(summarize_model(mpd, [[2, 1, 25600], [2, 1, 25600]], is_cuda = True, is_bi = True))
        print(summarize_model(msd, [[2, 1, 25600],[2, 1, 25600]], is_cuda = True, is_bi = True))

        os.makedirs(f"{a.ckpt_path}", exist_ok=True)
        print("checkpoints directory : ", f"{a.ckpt_path}")

    if os.path.isdir(f"{a.ckpt_path}"):
        cp_g = scan_checkpoint(f"{a.ckpt_path}", 'g_')
        cp_do = scan_checkpoint(f"{a.ckpt_path}", 'do_')


    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    # Optimizer
    optim_g = torch.optim.AdamW(generator.parameters(), cfg.train["lr"], betas=[0.6, cfg.train["beta2"]])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                cfg.train["lr"], betas=[0.6, cfg.train["beta2"]])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
    scheduler_g_warmup = WarmupScheduler(optim_g, warmup_steps=5, base_lr=cfg.train["lr"])
    scheduler_d_warmup = WarmupScheduler(optim_d, warmup_steps=5, base_lr=cfg.train["lr"])

    # Exponential weight decay
    scheduler_g_decay = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=cfg.train["weight_decay"], last_epoch=last_epoch)
    scheduler_d_decay = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=cfg.train["weight_decay"], last_epoch=last_epoch)



    # Train modes
    generator.train()
    mpd.train()
    msd.train()
    early_stopping = False
    # https://github.com/jik876/hifi-gan/blob/master/train.py
    for epoch in range(max(0, last_epoch), cfg.train["n_epoch"]):
        if rank == 0:
            start = time.time()
            print("Epoch: {} Learning Rate: {}".format(epoch + 1, optim_g.param_groups[0]["lr"]))



        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            y_high, y_low, _ = batch
            y_high = torch.autograd.Variable(y_high.to(device, non_blocking=True))
            y_low = torch.autograd.Variable(y_low.to(device, non_blocking=True))
            y_g_hat = generator(y_low)


            y_mel = torch.from_numpy(mel_spectrogram(y_high.squeeze(1), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"], cfg.audio["hop"],
                                    cfg.audio["window_size"], cfg.audio["fmin"], cfg.audio["fmax_for_loss"]))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))

            try:

                y_g_hat_mel = torch.from_numpy(mel_spectrogram(y_g_hat.squeeze(1), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"], cfg.audio["hop"],
                                       cfg.audio["window_size"], cfg.audio["fmin"], cfg.audio["fmax_for_loss"]))
            except:
                # For debugging
                print("error point is: ", y_g_hat_mel.shape)
                nan_mask = torch.isnan(y_g_hat)
                nan_indices = torch.nonzero(nan_mask)
                print("NaN indices:")
                print(nan_indices)
                break


            y_g_hat_mel = torch.autograd.Variable(y_g_hat_mel.to(device, non_blocking=True))



            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mstft = loss_fn(y_high, y_g_hat) * 10
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y_high, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y_high, y_g_hat)
            #loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            #loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            #loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_mstft
            loss_gen_all = loss_gen_s + loss_gen_f + loss_mel + loss_mstft
            loss_gen_all.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=3.0)
            optim_g.step()


            optim_d.zero_grad()
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y_high, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y_high, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            torch.nn.utils.clip_grad_norm_(mpd.parameters(), max_norm=2.0)
            torch.nn.utils.clip_grad_norm_(msd.parameters(), max_norm=2.0)
            optim_d.step()


            if rank == 0:
                if steps % cfg.train["stdout_interval"] == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))



                # wandb summary logging
                if steps % cfg.train["summary_interval"] == 0:
                    wandb.log({"training/gen_loss_total": loss_gen_all, "steps": steps})
                    wandb.log({"training/mel_spec_error": mel_error, "steps": steps})
                    wandb.log({"training/mpd_error": loss_disc_f, "steps": steps})
                    wandb.log({"training/msd_error": loss_disc_s, "steps": steps})
                    wandb.log({"training/generator_mpd_error": loss_gen_f, "steps": steps})
                    wandb.log({"training/generator_msd_error": loss_gen_s, "steps": steps})


                # Validation
                if steps % (cfg.train["validation_interval"]) == 0 and steps != 0:  # Validation every 2000 steps (Approx. 2.5 epochs in batch size of 64)
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):

                            y_high, y_low, _ = batch
                            y_high = y_high.to(device)

                            y_g_hat = generator(y_low.to(device))
                            y_mel = torch.from_numpy(mel_spectrogram(y_high.squeeze(1), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                                    cfg.audio["hop"],
                                                    cfg.audio["window_size"], cfg.audio["fmin"],
                                                    cfg.audio["fmax_for_loss"]))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking = True))
                            y_g_hat_mel = torch.from_numpy(mel_spectrogram(y_g_hat.squeeze(1), cfg.audio["nfft"], cfg.audio["n_mels"],
                                                          cfg.audio["sr"], cfg.audio["hop"],
                                                          cfg.audio["window_size"], cfg.audio["fmin"],
                                                          cfg.audio["fmax_for_loss"]))
                            y_g_hat_mel = torch.autograd.Variable(y_g_hat_mel.to(device, non_blocking=True))

                            loss_mel = F.l1_loss(y_mel, y_g_hat_mel)

                            val_err_tot += loss_mel


                        val_err = val_err_tot / (j + 1)
                        wandb.log({"validation/mel_error": val_err, "steps": steps})


                        # Audio logging
                        y_high_np = y_high.cpu().detach().numpy().astype(float).reshape(-1)
                        y_low_np = y_low.cpu().detach().numpy().astype(float).reshape(-1)
                        y_g_hat_np = y_g_hat.cpu().detach().numpy().astype(float).reshape(-1)

                        gt_high_audio = wandb.Audio(y_high_np, sample_rate=cfg.audio["sr"], caption="ground_truth_high_audio")
                        pred_audio = wandb.Audio(y_g_hat_np, sample_rate=cfg.audio["sr"], caption="predicted_audio")
                        gt_low_audio = wandb.Audio(y_low_np, sample_rate=cfg.audio["sr"], caption="ground_truth_low_audio")
                        wandb.log({f"ground truth high audio in epoch {epoch}": gt_high_audio})
                        wandb.log({f"ground truth low audio in epoch {epoch}": gt_low_audio})
                        wandb.log({f"predicted audio in epoch {epoch}": pred_audio})

                        # Mel Spectrogram logging
                        gt_high_mel = librosa.power_to_db(
                            mel_spectrogram(y_high.squeeze(), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                            cfg.audio["hop"],
                                            cfg.audio["window_size"], cfg.audio["fmin"],
                                            cfg.audio["fmax_for_loss"]), ref=np.max)
                        gt_low_mel = librosa.power_to_db(
                            mel_spectrogram(y_low.squeeze(), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                            cfg.audio["hop"],
                                            cfg.audio["window_size"], cfg.audio["fmin"],
                                            cfg.audio["fmax_for_loss"]), ref=np.max)
                        pred_mel = librosa.power_to_db(
                            mel_spectrogram(y_g_hat.squeeze(), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                            cfg.audio["hop"],
                                            cfg.audio["window_size"], cfg.audio["fmin"],
                                            cfg.audio["fmax_for_loss"]), ref=np.max)
                        gt_high_mel = wandb.Image(
                            librosa.display.specshow(gt_high_mel, sr=cfg.audio["sr"], x_axis='time', y_axis='mel',
                                                     fmax=24000), caption="gt_high_mel")
                        gt_low_mel = wandb.Image(librosa.display.specshow(gt_low_mel, sr=cfg.audio["sr"], x_axis='time',
                                                                          y_axis='mel', fmax=24000),
                                                 caption="gt_low_mel")
                        pred_mel = wandb.Image(librosa.display.specshow(pred_mel, sr=cfg.audio["sr"], x_axis='time',
                                                                        y_axis='mel', fmax=24000),
                                               caption="predicted_mel")

                        wandb.log({f"ground truth high mel in epoch {epoch},": gt_high_mel})
                        wandb.log({f"ground truth low mel in epoch {epoch}": gt_low_mel})
                        wandb.log({f"predicted mel in epoch {epoch}": pred_mel})

                        # Early stopping to prevent overfitting
                        if val_err < init_err:
                            early_stopping_count = 0
                            init_err = val_err
                        else:
                            print("Validation Error did not go down!")
                            early_stopping_count += 1

                        if early_stopping_count >= cfg.train["early_stopping_max"]:
                            print(f"Early stopping due to high validation error: \n {init_err} is the best validation error in the epoch: {epoch}")
                            early_stopping = True
                            break
                    generator.train()

                #checkpointing
                if steps % (cfg.train["checkpoint_interval"]) == 0 and steps != 0:
                    if best_err_for_ckpt > val_err:
                        best_err_for_ckpt = val_err
                        best_step = steps
                        checkpoint_path = "{}/g_{:08d}".format(a.ckpt_path, steps)
                        save_checkpoint(checkpoint_path,
                                        {'generator': generator.state_dict()})
                        checkpoint_path = "{}/do_{:08d}".format(a.ckpt_path, steps)
                        save_checkpoint(checkpoint_path,
                                        {'mpd': mpd.state_dict(),
                                         'msd': msd.state_dict(),
                                         'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                         'steps': steps,
                                         'epoch': epoch})
                        print(f"The error for steps {steps} is {best_err_for_ckpt}")
                    best_checkpoint_path = "{}/g_{:08d}".format(a.ckpt_path, best_step)



            steps += 1

        if early_stopping:
            print(f"Early Stopping because of overfitting, best checkpoint is : {best_checkpoint_path}")
            break
        if epoch < warmup_epoch:
            scheduler_g_warmup.step()
            scheduler_d_warmup.step()
        else:
            scheduler_g_decay.step()
            scheduler_d_decay.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default='project name')
    parser.add_argument('--ckpt_path', default="train_ckpt_path")

    a = parser.parse_args()
    with open("cfgs.json", "r") as file:

        json_config = json.load(file)
        hyperparameters = AttrDict(json_config)

    print('Initializing Training Process..')

    torch.manual_seed(hyperparameters.info["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hyperparameters.info["seed"])
        hyperparameters.train["n_gpu"] = torch.cuda.device_count()
        hyperparameters.train["batch_size"] = int(hyperparameters.train["batch_size"] / hyperparameters.train["n_gpu"])
        print('Batch size per GPU :',hyperparameters.train["batch_size"])
    else:
        pass


    world_size = 8
    mp.spawn(train, args=(hyperparameters, a), nprocs=2)

if __name__ == "__main__":
    main()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"

def cleanup():
    torch.distributed.destroy_process_group()
