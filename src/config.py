# training configuration
dataset_dir = '/Users/hongseokoh/Documents/GitHub/CSE203B-speech-enhancement/data/'
training_dataset_size = 1000
n_atoms = 512
alpha_val = 0.001
num_lasso_iterations = 1000
num_ksvd_iterations = 1
SNR_dB = 0
noise_type = 'white' # ['white', 'pink', 'brown', 'grey', 'blue', 'purple']

# signal processing configuration
frame_size = 1024
hop_size = 512
n_mels = 128
n_fft = 1024
sample_rate = 16000
duration = 1