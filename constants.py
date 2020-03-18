epochs = 100
num_samples = 30000
batch_size = 16
sample_rate = 16000

# STFT transform
lower_edge_hertz = 20.0
upper_edge_hertz = 8000.0
frame_length = 160
frame_step = frame_length // 4
num_fft = 2048
num_mel_filters = 512
