"""
PyTorch Datasets for TTS Training
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from .preprocessing import AudioPreprocessor
from text.phonemizer import Phonemizer


class TTSDataset(Dataset):
    """Dataset for acoustic model training"""
    
    def __init__(self, metadata_path, audio_dir, config, language='en', max_duration=10.0):
        """
        Initialize TTS dataset
        
        Args:
            metadata_path: Path to metadata CSV (format: filename|text|speaker_id)
            audio_dir: Directory containing audio files
            config: VocoderConfig object
            language: Language code
            max_duration: Maximum audio duration in seconds
        """
        self.audio_dir = audio_dir
        self.config = config
        self.language = language
        self.max_duration = max_duration
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path, sep='|', header=None,
                                   names=['filename', 'text', 'speaker_id'])
        
        # Initialize preprocessor and phonemizer
        self.preprocessor = AudioPreprocessor(config)
        self.phonemizer = Phonemizer(language)
        
        # Filter by duration
        self._filter_by_duration()
    
    def _filter_by_duration(self):
        """Filter out audio files that are too long"""
        max_samples = int(self.max_duration * self.config.sampling_rate)
        
        valid_indices = []
        for idx in range(len(self.metadata)):
            audio_path = os.path.join(self.audio_dir, self.metadata.iloc[idx]['filename'])
            if os.path.exists(audio_path):
                try:
                    audio, _ = self.preprocessor.load_audio(audio_path)
                    if len(audio) <= max_samples:
                        valid_indices.append(idx)
                except:
                    pass
        
        self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)
        print(f"Filtered dataset: {len(self.metadata)} samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            phonemes: (seq_len,) - phoneme IDs
            mel: (n_mel, time) - mel-spectrogram
            duration: (seq_len,) - phoneme durations
            pitch: (seq_len,) - pitch values
            energy: (seq_len,) - energy values
            speaker_id: int - speaker ID
        """
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = os.path.join(self.audio_dir, row['filename'])
        audio, _ = self.preprocessor.load_audio(audio_path)
        
        # Convert text to phonemes
        text = row['text']
        phoneme_ids = self.phonemizer.text_to_sequence(text)
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long)
        
        # Extract mel-spectrogram
        mel = self.preprocessor.compute_mel_spectrogram(audio)
        
        # Extract pitch and energy
        pitch = self.preprocessor.extract_pitch(audio)
        energy = self.preprocessor.extract_energy(audio)
        
        # Compute durations (simplified - in practice, use forced alignment)
        # Here we just divide mel frames evenly across phonemes
        mel_len = mel.size(1)
        phoneme_len = len(phoneme_ids)
        durations = self._compute_durations(phoneme_len, mel_len)
        
        # Align pitch and energy to phoneme level
        pitch_phoneme = self._align_to_phonemes(pitch, durations)
        energy_phoneme = self._align_to_phonemes(energy, durations)
        
        # Speaker ID
        speaker_id = int(row['speaker_id']) if 'speaker_id' in row else 0
        
        return {
            'phonemes': phoneme_ids,
            'mel': mel,
            'durations': durations,
            'pitch': pitch_phoneme,
            'energy': energy_phoneme,
            'speaker_id': speaker_id
        }
    
    def _compute_durations(self, phoneme_len, mel_len):
        """Compute phoneme durations (simplified)"""
        # Evenly distribute mel frames across phonemes
        base_duration = mel_len // phoneme_len
        remainder = mel_len % phoneme_len
        
        durations = torch.full((phoneme_len,), base_duration, dtype=torch.long)
        durations[:remainder] += 1
        
        return durations
    
    def _align_to_phonemes(self, frame_values, durations):
        """Align frame-level values to phoneme level"""
        phoneme_values = []
        pos = 0
        
        for dur in durations:
            dur = int(dur.item())
            if dur > 0:
                # Average over duration
                phoneme_val = frame_values[pos:pos+dur].mean()
                phoneme_values.append(phoneme_val)
                pos += dur
            else:
                phoneme_values.append(torch.tensor(0.0))
        
        return torch.stack(phoneme_values)


class VocoderDataset(Dataset):
    """Dataset for vocoder training"""
    
    def __init__(self, metadata_path, audio_dir, config, segment_size=None):
        """
        Initialize vocoder dataset
        
        Args:
            metadata_path: Path to metadata CSV
            audio_dir: Directory containing audio files
            config: VocoderConfig object
            segment_size: Audio segment size for training (None = full audio)
        """
        self.audio_dir = audio_dir
        self.config = config
        self.segment_size = segment_size if segment_size else config.segment_size
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path, sep='|', header=None,
                                   names=['filename', 'text', 'speaker_id'])
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(config)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            mel: (n_mel, time) - mel-spectrogram
            audio: (time * hop_length,) - audio waveform
        """
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = os.path.join(self.audio_dir, row['filename'])
        audio, _ = self.preprocessor.load_audio(audio_path)
        
        # Extract mel-spectrogram
        mel = self.preprocessor.compute_mel_spectrogram(audio)
        
        # Convert to tensor
        audio = torch.from_numpy(audio).float()
        
        # Random segment for training
        if self.segment_size is not None and len(audio) > self.segment_size:
            mel_segment_size = self.segment_size // self.config.hop_length
            
            if mel.size(1) > mel_segment_size:
                start_frame = np.random.randint(0, mel.size(1) - mel_segment_size)
                mel = mel[:, start_frame:start_frame + mel_segment_size]
                
                start_sample = start_frame * self.config.hop_length
                audio = audio[start_sample:start_sample + self.segment_size]
        
        return {
            'mel': mel,
            'audio': audio
        }


def collate_fn_tts(batch):
    """Collate function for TTS dataset"""
    # Find max lengths
    max_phoneme_len = max([item['phonemes'].size(0) for item in batch])
    max_mel_len = max([item['mel'].size(1) for item in batch])
    
    batch_size = len(batch)
    n_mel = batch[0]['mel'].size(0)
    
    # Initialize tensors
    phonemes_padded = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    mels_padded = torch.zeros(batch_size, n_mel, max_mel_len)
    durations_padded = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    pitch_padded = torch.zeros(batch_size, max_phoneme_len)
    energy_padded = torch.zeros(batch_size, max_phoneme_len)
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)
    
    phoneme_lengths = torch.zeros(batch_size, dtype=torch.long)
    mel_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        phoneme_len = item['phonemes'].size(0)
        mel_len = item['mel'].size(1)
        
        phonemes_padded[i, :phoneme_len] = item['phonemes']
        mels_padded[i, :, :mel_len] = item['mel']
        durations_padded[i, :phoneme_len] = item['durations']
        pitch_padded[i, :phoneme_len] = item['pitch']
        energy_padded[i, :phoneme_len] = item['energy']
        speaker_ids[i] = item['speaker_id']
        
        phoneme_lengths[i] = phoneme_len
        mel_lengths[i] = mel_len
    
    return {
        'phonemes': phonemes_padded,
        'mels': mels_padded,
        'durations': durations_padded,
        'pitch': pitch_padded,
        'energy': energy_padded,
        'speaker_ids': speaker_ids,
        'phoneme_lengths': phoneme_lengths,
        'mel_lengths': mel_lengths
    }


def collate_fn_vocoder(batch):
    """Collate function for vocoder dataset"""
    # Find max mel length
    max_mel_len = max([item['mel'].size(1) for item in batch])
    max_audio_len = max([item['audio'].size(0) for item in batch])
    
    batch_size = len(batch)
    n_mel = batch[0]['mel'].size(0)
    
    # Initialize tensors
    mels_padded = torch.zeros(batch_size, n_mel, max_mel_len)
    audios_padded = torch.zeros(batch_size, max_audio_len)
    
    # Fill tensors
    for i, item in enumerate(batch):
        mel_len = item['mel'].size(1)
        audio_len = item['audio'].size(0)
        
        mels_padded[i, :, :mel_len] = item['mel']
        audios_padded[i, :audio_len] = item['audio']
    
    return {
        'mels': mels_padded,
        'audios': audios_padded
    }
