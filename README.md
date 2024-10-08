# Conv-TasNet-visualization
A implementation of [Conv-TasNet](https://arxiv.org/abs/1809.07454) visualization.

[Conv-TasNet-visualization](https://github.com/SXKA/Conv-TasNet-visualization) use [Asteroid](https://github.com/asteroid-team/asteroid) model.

The images of visualization are produced with [JorisCos/ConvTasNet_Libri2Mix_sepclean_8k](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_8k).
## Usage
```powershell
python visualization.py --model_path conv_tasnet_model_path --mixture_path mixture_audio_path --spk1_path spk1_audio_path --spk2_path spk2_audio_path
```
where:
- ``--model_path`` is Conv-TasNet model path.
- ``--mixture_path`` is mixture audio file path.
- ``--spk1_path`` is speaker 1 audio file path.
- ``--spk2_path`` is speaker 2 audio file path.
## Visualization
The sample audio isn't provided because it is generated from [WSJ0](https://catalog.ldc.upenn.edu/LDC93S6A).
### Waveform
![mixture_waveform](https://github.com/SXKA/Conv-TasNet-visualization/blob/master/png/mixture_waveform.png)
![spk1_waveform](https://github.com/SXKA/Conv-TasNet-visualization/blob/master/png/spk1_waveform.png)
![spk2_waveform](https://github.com/SXKA/Conv-TasNet-visualization/blob/master/png/spk2_waveform.png)
### Encoder and decoder basis functions
<div align="center">
  <img src="https://github.com/SXKA/Conv-TasNet-visualization/blob/master/png/encoder_weights.png" alt="encoder_weights" width="400" height="800" />&nbsp;
  <img src="https://github.com/SXKA/Conv-TasNet-visualization/blob/master/png/encoder_weights_fft.png" alt="encoder_weights_fft" width="400" height="800" />
  <img src="https://github.com/SXKA/Conv-TasNet-visualization/blob/master/png/decoder_weights.png" alt="decoder_weights" width="400" height="800" />&nbsp;
  <img src="https://github.com/SXKA/Conv-TasNet-visualization/blob/master/png/decoder_weights_fft.png" alt="decoder_weights_fft" width="400" height="800" />
</div>

### Encoder representation
![encoder_output](https://github.com/SXKA/Conv-TasNet-visualization/blob/master/png/encoder_output.png)
### Source masks
![spk1_mask](https://github.com/SXKA/Conv-TasNet-visualization/blob/master/png/spk1_mask.png)
![spk2_mask](https://github.com/SXKA/Conv-TasNet-visualization/blob/master/png/spk2_mask.png)
