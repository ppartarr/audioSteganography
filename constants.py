# TRAINING
epochs = 200
num_samples = 6300
batch_size = 64
data_dir = "data"

# PARSING
# sample rate
sample_rate = 16000
# number of Fast Fourier transform
n_fft = 512
# sliding window 10ms: int(np.ceil(0.010 * sample_rate))
win_length = 160
# window overlapping 25%
hop_length = win_length // 4
# maximum frequency
fmax = 8000.0
# number of mel filters
n_mels = 64
