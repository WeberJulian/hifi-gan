import math
import os
import random
import torch
import glob
import soundfile as sf
import torch.utils.data
import numpy as np
import librosa
import torchaudio
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from audiomentations import (
    Compose,
    AddGaussianSNR,
    ApplyImpulseResponse,
    Mp3Compression,
    SevenBandParametricEQ,
    AddBackgroundNoise,
)

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    return sf.read(full_path)


def _rms_norm(wav, db_level=-27):
    r = 10 ** (db_level / 20)
    a = np.sqrt((len(wav) * (r ** 2)) / np.sum(wav ** 2))
    return wav * a


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(0),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze()

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    all_files = glob.glob(
        os.path.join(a.input_wavs_dir, "**", "*.flac"), recursive=True
    )
    all_files += glob.glob(
        os.path.join(a.input_wavs_dir, "**", "*.wav"), recursive=True
    )
    assert len(all_files) > 100, "Not enough files in the dataset"
    return all_files[:-100], all_files[-100:]


class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        training_files,
        segment_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
    ):
        self.config = config
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.resample_down = torchaudio.transforms.Resample(self.config.sampling_rate, self.config.sampling_rate // 3)
        if self.augment:
            self.augmentor = Compose(
                [
                    AddBackgroundNoise(config.noise_path, max_snr_in_db=55, min_snr_in_db=30, p=0.5),
                    ApplyImpulseResponse(config.rir_path, leave_length_unchanged=True, p=0.666),
                    AddGaussianSNR(min_snr_in_db=30, max_snr_in_db=60, p=0.666),
                    SevenBandParametricEQ(p=0.666, min_gain_db=-4.0, max_gain_db=4.0),
                    Mp3Compression(backend="pydub", max_bitrate=64, min_bitrate=16, p=0.1),
                ]
            )

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = _rms_norm(audio)
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    "{} SR doesn't match target {} SR".format(
                        sampling_rate, self.sampling_rate
                    )
                )
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start : audio_start + self.segment_size]
            else:
                audio = torch.nn.functional.pad(
                    audio, (0, self.segment_size - audio.size(1)), "constant"
                )
        audio = audio.squeeze()
        if len(audio) %3 != 0:
            audio = audio[:-(len(audio) % 3)]

        augmented_audio = self.resample_down(audio)

        if self.config.augment:
            augmented_audio = self.augment(augmented_audio)


        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_loss,
            center=False,
        )

        return (augmented_audio, audio, filename, mel)

    def __len__(self):
        return len(self.audio_files)

    def augment(self, augmented_audio, sampling_rate=16000):
        augmented_audio = augmented_audio.cpu().float().numpy()
        augmented_audio = self.augmentor(
            samples=augmented_audio,
            sample_rate=sampling_rate,
        )
        augmented_audio = _rms_norm(augmented_audio)
        augmented_audio = torch.FloatTensor(augmented_audio).squeeze()
        return augmented_audio
