import argparse

from argparse import Namespace

import librosa
import numpy as np
import seaborn as sns
import seaborn_image as isns
import soundfile as sf
import torch
import torchaudio

from asteroid.models import ConvTasNet
from asteroid.losses.sdr import singlesrc_neg_sisdr
from asteroid.utils.torch_utils import pad_x_to_y
from matplotlib.axes import Axes
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.fft import rfft, rfftfreq


@torch.inference_mode()
def main(args: Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvTasNet.from_pretrained(args.model_path).to(device)
    sample_rate = int(model.sample_rate)

    encoder_state_dict = model.encoder.state_dict()
    encoder_weights = encoder_state_dict["filterbank._filters"].cpu().numpy().squeeze()
    z = linkage(
        encoder_weights, method="average", metric="euclidean", optimal_ordering=True
    )
    sorted_indices = leaves_list(z)
    encoder_weights = encoder_weights[sorted_indices]

    ax = weightsplot(encoder_weights, sample_rate)
    ax.set_title("Encoder weights")
    ax.get_figure().savefig("encoder_weights.png", bbox_inches="tight")

    ax = weightsfftplot(encoder_weights, sample_rate)
    ax.set_title("Encoder weights ||FFT||")
    figure = ax.get_figure()
    figure.savefig("encoder_weights_fft.png", bbox_inches="tight")
    figure.clear()

    spk1 = librosa.load(args.spk1_path, sr=sample_rate)[0]
    spk2 = librosa.load(args.spk2_path, sr=sample_rate)[0]

    sns.lineplot(
        x=np.arange(len(spk1)) / sample_rate,
        y=spk1,
        alpha=0.8,
        color=sns.color_palette("tab10")[0],
        label="spk1",
    )
    ax = sns.lineplot(
        x=np.arange(len(spk1)) / sample_rate,
        y=spk2,
        alpha=0.8,
        color=sns.color_palette("tab10")[3],
        label="spk2",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Mixture waveform")
    ax.get_figure().savefig("mixture_waveform.png", bbox_inches="tight")

    mixture, sr = torchaudio.load(args.mixture_path)
    mixture = torchaudio.functional.resample(
        mixture, orig_freq=sr, new_freq=sample_rate
    )
    encoder_output = model.forward_encoder(mixture.unsqueeze(1).to(device))
    encoder_output_numpy = encoder_output.squeeze().cpu().numpy()

    encoder_spk1_output = model.forward_encoder(torch.from_numpy(spk1).to(device))
    encoder_spk2_output = model.forward_encoder(torch.from_numpy(spk2).to(device))
    encoder_spk1_output_numpy = encoder_spk1_output.cpu().numpy()
    encoder_spk2_output_numpy = encoder_spk2_output.cpu().numpy()
    power_coef = np.ones_like(encoder_spk1_output_numpy)
    power_coef[
        np.abs(encoder_spk1_output_numpy) > np.abs(encoder_spk2_output_numpy)
    ] = -1
    power_coef[
        np.isclose(np.abs(encoder_spk1_output_numpy), np.abs(encoder_spk2_output_numpy))
    ] = 0

    encoder_output_numpy = power_coef * np.abs(encoder_output_numpy)

    ax = isns.imgplot(
        encoder_output_numpy,
        cmap="vlag",
        robust=True,
        cbar=False,
        showticks=True,
    )
    ax.set_aspect(5)
    ax.set_xlabel("Time (ms)")
    ax.set_xticks(
        (0, encoder_output_numpy.shape[1] - 1),
        (0, int(1000 * mixture.shape[1] / sample_rate)),
    )
    ax.set_yticks(
        (0, encoder_output_numpy.shape[0] - 1), (encoder_output_numpy.shape[0], 1)
    )
    ax.set_title("Encoder output")
    ax.get_figure().savefig("encoder_output.png", bbox_inches="tight")

    masks = model.forward_masker(encoder_output)
    spk1_mask, spk2_mask = masks.squeeze().cpu().numpy()

    ax = isns.imgplot(
        spk1_mask,
        gray=True,
        robust=True,
        cbar=False,
        showticks=True,
    )
    ax.set_aspect(5)
    ax.set_xlabel("Time (ms)")
    ax.set_xticks(
        (0, spk1_mask.shape[1] - 1),
        (0, int(1000 * mixture.shape[1] / sample_rate)),
    )
    ax.set_ylabel("Basis index")
    ax.set_yticks((0, spk1_mask.shape[0] - 1), (spk1_mask.shape[0], 1))
    ax.set_title("Spk1 mask")
    ax.get_figure().savefig("spk1_mask.png", bbox_inches="tight")

    ax = isns.imgplot(
        spk2_mask,
        gray=True,
        robust=True,
        cbar=False,
        showticks=True,
    )
    ax.set_aspect(5)
    ax.set_xlabel("Time (ms)")
    ax.set_xticks(
        (0, spk1_mask.shape[1] - 1),
        (0, int(1000 * mixture.shape[1] / sample_rate)),
    )
    ax.set_ylabel("Basis index")
    ax.set_yticks((0, spk2_mask.shape[0] - 1), (spk2_mask.shape[0], 1))
    ax.set_title("Spk2 mask")
    ax.get_figure().savefig("spk2_mask.png", bbox_inches="tight")

    decoder_state_dict = model.decoder.state_dict()
    decoder_weights = decoder_state_dict["filterbank._filters"].squeeze().cpu().numpy()
    z = linkage(
        decoder_weights, method="average", metric="euclidean", optimal_ordering=True
    )
    sorted_indices = leaves_list(z)
    decoder_weights = decoder_weights[sorted_indices]

    ax = weightsplot(decoder_weights, sample_rate)
    ax.set_title("Decoder weights")
    ax.get_figure().savefig("decoder_weights.png", bbox_inches="tight")

    ax = weightsfftplot(decoder_weights, sample_rate)
    ax.set_title("Decoder weights ||FFT||")
    figure = ax.get_figure()
    figure.savefig("decoder_weights_fft.png", bbox_inches="tight")
    figure.clear()

    masked_output = model.apply_masks(encoder_output, masks)
    decoder_output = model.forward_decoder(masked_output)
    waveforms = pad_x_to_y(decoder_output, mixture.unsqueeze(1)).squeeze()
    waveforms *= mixture.abs().sum() / waveforms.abs().sum()
    spk1_waveform, spk2_waveform = waveforms.cpu().numpy()
    if singlesrc_neg_sisdr(
        torch.from_numpy(spk1_waveform).unsqueeze(0),
        torch.from_numpy(spk1).unsqueeze(0),
    ) > singlesrc_neg_sisdr(
        torch.from_numpy(spk2_waveform).unsqueeze(0),
        torch.from_numpy(spk1).unsqueeze(0),
    ):
        spk1_waveform, spk2_waveform = spk2_waveform, spk1_waveform

    sf.write("spk1_estimate.wav", spk1_waveform, sample_rate)
    sf.write("spk2_estimate.wav", spk2_waveform, sample_rate)

    ax = sns.lineplot(
        x=np.arange(len(spk1_waveform)) / sample_rate,
        y=spk1_waveform,
        color=sns.color_palette("tab10")[0],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Spk1 waveform")
    figure = ax.get_figure()
    figure.savefig("spk1_waveform.png", bbox_inches="tight")
    figure.clear()

    ax = sns.lineplot(
        x=np.arange(len(spk2_waveform)) / sample_rate,
        y=spk2_waveform,
        color=sns.color_palette("tab10")[3],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Spk2 waveform")
    ax.get_figure().savefig("spk2_waveform.png", bbox_inches="tight")


def weightsplot(weights: np.ndarray, sample_rate: int) -> Axes:
    """Plot weights as 2-D image.

    Args:
        weights (np.ndarray): Encoder/Decoder weights
        sample_rate (int): Sample rate

    Returns:
        Axes: Matplotlib axes where the image is drawn.
    """
    ax = isns.imgplot(
        weights,
        cmap="vlag",
        robust=True,
        cbar=False,
        showticks=True,
    )
    ax.figure.colorbar(ax.images[0], ax=ax, shrink=0.5, pad=0.01, anchor=(0.0, 0.0))
    ax.set_aspect(0.15)
    ax.set_xlabel("Time (ms)")
    ax.set_xticks(
        (0, weights.shape[1] - 1),
        (0, int(1000 * weights.shape[1] / sample_rate)),
    )
    ax.set_ylabel("Basis index")
    ax.set_yticks((0, weights.shape[0] - 1), (weights.shape[0], 0))
    return ax


def weightsfftplot(weights: np.ndarray, sample_rate: int) -> Axes:
    """Perform and visualize FFT of weights.

    Args:
        weights (np.ndarray): Encoder/Decoder weights
        sample_rate (int): Sample rate

    Returns:
        Axes: Matplotlib axes where the image is drawn.
    """
    freq = (rfftfreq(weights.shape[1]) * sample_rate / 1000).astype(int)

    weights_fft = np.abs(rfft(weights, axis=1))
    z = linkage(
        weights_fft, method="average", metric="euclidean", optimal_ordering=True
    )
    sorted_indices = leaves_list(z)
    weights_fft = weights_fft[sorted_indices]
    ax = isns.imgplot(
        weights_fft,
        cmap="gray_r",
        robust=True,
        cbar=False,
        showticks=True,
    )
    ax.set_aspect(0.075)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_xticks((0, weights_fft.shape[1] - 1), (freq[0], freq[-1]))
    ax.set_ylabel("Basis index")
    ax.set_yticks((0, weights_fft.shape[0] - 1), (weights_fft.shape[0], 1))
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="pytorch_model.bin",
        help="Conv-TasNet model path",
    )
    parser.add_argument(
        "--mixture_path",
        type=str,
        default="mixture.wav",
        help="Mixture audio file path",
    )
    parser.add_argument(
        "--spk1_path", type=str, default="spk1.wav", help="Speaker 1 audio file path"
    )
    parser.add_argument(
        "--spk2_path", type=str, default="spk2.wav", help="Speaker 2 audio file path"
    )
    main(parser.parse_args())
