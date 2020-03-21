import matplotlib.pyplot as plt
import utils
import librosa
import librosa.display
import numpy as np
import os

# config
output_dir = 'plots'
data_file = 'data/TRAIN/DR1/FCJF0/SA1.WAV.wav'

os.makedirs(output_dir, exist_ok=True)

y, sr = librosa.load(data_file, duration=5)

stft = librosa.stft(y, n_fft=512, hop_length=512 // 4)

stft_specgram = librosa.power_to_db(
    np.abs(stft)**2,
    ref=np.max)

mel_specgram = librosa.power_to_db(
    librosa.feature.melspectrogram(y, sr=sr, n_mels=128, fmax=8000),
    ref=np.max)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.waveplot(y, sr=sr, x_axis=None)
plt.ylabel('Amplitude')
plt.title('Waveform representation of a WAV audio stream')

plt.subplot(3, 1, 2)
librosa.display.specshow(stft_specgram, sr=sr, y_axis='log')
plt.ylabel('Hz')
plt.title('STFT spectrogram of a WAV audio stream')
# plt.colorbar(format='%+2.0f dB')

plt.subplot(3, 1, 3)
plt.title('Mel-frequency spectrogram of a WAV audio stream')
librosa.display.specshow(mel_specgram, sr=sr,
                         x_axis='time', y_axis='log')
plt.ylabel('Hz')
plt.xlabel('Time in s')
# plt.colorbar(format='%+2.0f dB')
x_axis = 'time',
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(output_dir, 'wav-stft-mel'))
