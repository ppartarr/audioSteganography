epochs = 300
num_samples = 4620
batch_size = 64
sample_rate = 16000
audio_length = 2.5

# STFT transform
lower_edge_hertz = 20.0
upper_edge_hertz = 8000.0
frame_length = 160
frame_step = frame_length // 4
num_mel_filters = 512
