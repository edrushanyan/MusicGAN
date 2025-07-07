# MusicGAN: LSTM-based WGAN for Piano Music Generation

This project implements a Generative Adversarial Network (GAN) for symbolic piano music generation using MIDI data. The architecture is based on the Wasserstein GAN (WGAN) framework, with both the generator and discriminator designed using LSTM networks. The model is trained on two large-scale piano MIDI datasets: MAESTRO and GiantMIDI-Piano.

## Overview

The objective of this project is to explore the application of adversarial training to sequential music data, aiming to generate coherent, expressive piano performances. The model leverages recurrent neural networks to capture the temporal structure of musical sequences and uses adversarial feedback to improve output realism.

## Architecture

- **Model Type**: Wasserstein GAN (WGAN)
- **Generator**: LSTM-based network
- **Discriminator**: LSTM-based network
- **Loss Function**: Wasserstein loss
- **Optimizer**: Adam
- **Epochs Trained**: 105 epochs

## Datasets

- **MAESTRO Dataset**: Dataset composed of about 200 hours of virtuosic piano performances captured with fine alignment (~3 ms) between note labels and audio waveforms.
- **GiantMIDI-Piano**:  A classical piano MIDI dataset contains 10,855 MIDI files of 2,786 composers. GiantMIDI-Piano are transcribed from live recordings with a high-resolution piano transcription system

Datasets are used in MIDI or MID form, and preprocessing includes conversion to piano roll representations.

## Results

The model was trained for 105 epochs. While initial training produced musically plausible outputs, signs of overfitting were observed.

## Limitations and Future Work

Current limitations include:
- Overfitting due to insufficient regularization
- Lack of structural diversity in generated outputs
- Evaluation metrics for musical quality are limited and subjective

Planned improvements:
- Introduce dropout or other regularization techniques
- Incorporate event-based representations and improve evaluation methodology