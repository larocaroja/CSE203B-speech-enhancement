# training configuration
dataset_dir = '/data/librispeech/CSE203B-speech-enhancement/data/'
train_dataset_size = -1 # whole: -1
valid_dataset_size = 250 # whole: -1
n_atoms = 256 # [128, 256]
alpha_val = 0.001
num_lasso_iterations = 1000
num_ksvd_iterations = 10
SNR_dB = 0
noise_type = 'white' # ['white', 'pink', 'brown', 'grey', 'blue', 'purple']

# signal processing configuration
frame_size = 1024
hop_size = 512
n_mels = 128
n_fft = 1024
sample_rate = 16000
duration = 1