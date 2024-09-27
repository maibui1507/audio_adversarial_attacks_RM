from base import BaseAugmentor
from utils import librosa_to_pydub
import random
import soundfile as sf
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

class CopyPasteFrequencyAugmentor(BaseAugmentor):
    """
    Copy and Paste augmentation in the frequency domain.
    Config:
        shuffle_ratio (float): The ratio of frames to shuffle (between 0 and 1).
        frame_size (int): The size of each frame in the frequency domain.
                          If zero, the frame size is randomly selected from [10, 40] frames.
    """
    def __init__(self, config: dict):
        """
        Initialize the `CopyPasteAugmentor` object.
        
        :param config: dict, configuration dictionary
        """
        super().__init__(config)
        self.shuffle_ratio = config['shuffle_ratio']
        self.frame_size = config['frame_size']
        assert 0.0 < self.shuffle_ratio < 1.0
        assert self.frame_size >= 0
        # Determine frame size
        if self.frame_size == 0:
            self.frame_size = np.random.randint(10, 40)  # Randomly select frame size from [10, 40] frames

    def transform(self):
        """
        Transform the audio by shuffling frames in the frequency domain.
        """
        # Perform STFT
        D = librosa.stft(self.data)
        magnitude, phase = np.abs(D), np.angle(D)
        
        # Split magnitude spectrogram into frames
        num_frames = magnitude.shape[1] // self.frame_size
        frames = [magnitude[:, i*self.frame_size:(i+1)*self.frame_size] for i in range(num_frames)]
        
        # Handle the last chunk of frames if it's not a full frame
        if magnitude.shape[1] % self.frame_size != 0:
            frames.append(magnitude[:, num_frames*self.frame_size:])
        
        # Determine the number of frames to shuffle
        num_frames_to_shuffle = int(len(frames) * self.shuffle_ratio)
        
        # Randomly shuffle a ratio of frames
        if num_frames_to_shuffle > 0:
            shuffle_indices = np.random.choice(len(frames), num_frames_to_shuffle, replace=False)
            shuffled_frames = np.array(frames, dtype=object)[shuffle_indices]
            np.random.shuffle(shuffled_frames)
            for i, idx in enumerate(shuffle_indices):
                frames[idx] = shuffled_frames[i]
        
        # Concatenate frames back together
        transformed_magnitude = np.concatenate(frames, axis=1)
        
        # Combine magnitude with original phase
        transformed_D = transformed_magnitude * np.exp(1j * phase)
        
        # Perform inverse STFT
        transformed_audio = librosa.istft(transformed_D)
        
        # Transform to pydub audio segment
        self.augmented_audio = librosa_to_pydub(transformed_audio, sr=self.sr)
