import pretty_midi as pm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import multiprocessing
import glob


MAX_LENGTH = 512
FS = 100

# Converting MIDI to Tensors with fixed length
def midi_to_piano_roll(midi_file, max_length=MAX_LENGTH, fs=FS):
    '''Convert MIDI file to fixed-length piano roll'''
    midi_data = pm.PrettyMIDI(midi_file)
    piano_roll = midi_data.get_piano_roll(fs=fs)[:, :max_length]  # Trim
            
    # Pad if shorter than max_length
    if piano_roll.shape[1] < max_length:
        pad_size = max_length - piano_roll.shape[1]
        piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_size)), 'constant')
                
    return piano_roll


def normalize_piano_roll(piano_roll):
    '''Normalize piano roll values between 0 and 1'''
    return np.clip(piano_roll / 127.0, 0, 1)

def midi_to_tensor(midi_file, max_length=MAX_LENGTH, fs=FS):
    '''Convert MIDI file to properly shaped tensor [max_length, 128]'''
    piano_roll = midi_to_piano_roll(midi_file, max_length, fs)
    if piano_roll is None:
        return None
    return torch.tensor(normalize_piano_roll(piano_roll.T), dtype=torch.float32)

# Implementation of Dataset Class and Dataloader
class MIDIDataset(Dataset):
    def __init__(self, dataset_path, max_length=MAX_LENGTH, fs=FS):
        '''Initialize dataset with configurable parameters'''
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.fs = fs
        self.midi_files = [os.path.join(dataset_path, i) for i in os.listdir(dataset_path) if i.endswith('.midi') or i.endswith('.mid')]
        
        print(f"Found {len(self.midi_files)} MIDI files")

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, index):
        midi_file = self.midi_files[index]
        tensor = midi_to_tensor(midi_file, self.max_length, self.fs)
        if tensor is None:  # If file failed to load
            return torch.zeros((self.max_length, 128))  # Silent fallback
        return tensor

dataset_path = r'C:\Users\ASUS\Desktop\Projects\MusicGAN\midi_dataset'
midi_dataset = MIDIDataset(dataset_path)
midi_dataloader = DataLoader(
                            midi_dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=0,
                            )

# Implementation of LSTM Generator
class LSTMGenerator(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=128, output_dim=128, num_layers=2):
        super(LSTMGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
                            input_size=latent_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True
                            )
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        lstm_output, _ = self.lstm(x, (h_0, c_0))
        output = self.tanh(self.fc(lstm_output))
        return output

# Implementation of LSTM Discriminator
class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2):
        super(LSTMDiscriminator, self).__init__()
        self.lstm = nn.LSTM(
                            input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True
                            )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        
        lstm_output, _ = self.lstm(x, (h_0, c_0))
        lstm_hidden = lstm_output[:, -1, :]
        return self.fc(lstm_hidden)

# Implementation of loss function and clip discriminator weights
def generator_loss(fake_scores):
    return -torch.mean(fake_scores)

def discriminator_loss(real_scores, fake_scores):
    return -torch.mean(real_scores) + torch.mean(fake_scores)

def clip_discriminator_weights(discriminator, clip_value=0.01):
    for p in discriminator.parameters():
        p.data.clamp_(-clip_value, clip_value)

# Function for saving generated MIDI files
def save_generated_midi(tensor, file_name, fs=100):
    if tensor.ndim == 3:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().numpy()
    piano_roll = np.clip(tensor * 127, 0, 127).astype(np.uint8)

    midi = pm.PrettyMIDI()
    instrument = pm.Instrument(program=0)

    for pitch in range(piano_roll.shape[1]):
        note_on = False
        start = 0
        for t in range(piano_roll.shape[0]):
            velocity = piano_roll[t, pitch]
            if velocity > 0 and not note_on:
                note_on = True
                start = t
            elif velocity == 0 and note_on:
                note_on = False
                end = t
                note = pm.Note(
                                velocity=int(np.max(piano_roll[start:end, pitch])),
                                pitch=pitch,
                                start=start / fs,
                                end=end / fs
                                )
                instrument.notes.append(note)

        if note_on:
            note = pm.Note(
                            velocity=int(np.max(piano_roll[start:, pitch])),
                            pitch=pitch,
                            start=start / fs,
                            end=piano_roll.shape[0] / fs
                            )
            instrument.notes.append(note)
    midi.instruments.append(instrument)
    midi.write(file_name)

def main():
    output_dir = r'C:\Users\ASUS\Desktop\Projects\MusicGAN\generated_midi'
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    midi_dir = os.path.join(output_dir, 'midis')
    os.makedirs(midi_dir, exist_ok=True)

    device = torch.device('cuda')

    # Model parameters
    latent_dim = 128
    hidden_dim = 128
    output_dim = 128
    num_epochs = 200
    n_critic = 5
    clip_value = 0.01

    generator = LSTMGenerator(latent_dim, hidden_dim, output_dim).to(device)
    discriminator = LSTMDiscriminator().to(device)

    # Optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=5e-5, betas=(0.5, 0.9))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=5e-5, betas=(0.5, 0.9))

    # Training loop
    for epoch in range(num_epochs):
        for i, real_batch in enumerate(midi_dataloader):
                
            real_batch = real_batch.to(device)
            
            # Train discriminator
            for _ in range(n_critic):
                z = torch.randn(real_batch.size(0), real_batch.size(1), latent_dim).to(device)
                fake_batch = generator(z).detach()

                disc_optimizer.zero_grad()
                real_scores = discriminator(real_batch)
                fake_scores = discriminator(fake_batch)
                d_loss = discriminator_loss(real_scores, fake_scores)
                d_loss.backward()
                disc_optimizer.step()
                clip_discriminator_weights(discriminator, clip_value)

            # Train generator
            z = torch.randn(real_batch.size(0), real_batch.size(1), latent_dim).to(device)
            gen_optimizer.zero_grad()
            generated_batch = generator(z)
            fake_scores = discriminator(generated_batch)
            g_loss = generator_loss(fake_scores)
            g_loss.backward()
            gen_optimizer.step()

            # Logging
            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(midi_dataloader)}] "
                      f"D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

        # Save checkpoint and samples
        if epoch % 5 == 0:
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch_{epoch+1}.pt'))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch+1}.pt'))

        with torch.no_grad():
            sample_z = torch.randn(1, MAX_LENGTH, latent_dim).to(device)
            sample_output = generator(sample_z)[0]
            midi_path = os.path.join(midi_dir, f'sample_epoch_{epoch + 1}.midi')
            save_generated_midi(sample_output, file_name=midi_path)

if __name__ == "__main__":
    main()