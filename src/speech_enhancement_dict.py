import sys
import os

from glob import glob
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.decomposition import DictionaryLearning, NMF
import librosa
import librosa.display
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

import config
import pesq


class SpeechEnhancement:
    def __init__(self, config):
        self.config = config
    
    def lasso_sparse_coding(self, D, b, alpha_val):
        """
        Solve the sparse coding problem using Lasso.
        D : dictionary (n_mels, n_atoms)
        b : input signal (n_mels,)
        alpha_val : regularization parameter
        """
        lasso = Lasso(alpha=alpha_val, fit_intercept=False, max_iter=config.num_lasso_iterations, tol=1e-4, positive=True)
        lasso.fit(D, b)
        # print(D.shape, b.shape)
        # print(lasso.coef_.shape)
        # print(lasso.coef_)
        # print(lasso.dual_gap_)
        # print(b)
        # print(D @ lasso.coef_)
        return lasso.coef_

        # sys.exit()
    
    def k_svd(self, B):
        """
        K-SVD algorithm for dictionary learning, using Lasso for sparse coding.
        """
        m, n = B.shape # (n_mels, n_samples * n_frames)
        print(f"Number of samples: {n}, Number of mel bins: {m}")

        # Initialize dictionary with random atoms
        D = np.random.rand(m, self.config.n_atoms)
        X = np.zeros((self.config.n_atoms, n))

        for iteration in tqdm(range(self.config.num_ksvd_iterations)):
            # Sparse coding step with Lasso
            # for i in range(n):
            #     X[:, i] = self.lasso_sparse_coding(D, B[:, i], self.config.alpha_val)
            # print(D.shape, B.shape)
            X = self.lasso_sparse_coding(D, B, self.config.alpha_val)
            X = X.T
            # print(X.shape)
            print(f"Iteration {iteration+1}/{self.config.num_ksvd_iterations} complete.")

            # Dictionary update step using simplified K-SVD update algorithm
            for j in range(self.config.n_atoms):
                # print(B.shape, D.shape, X.shape)
                # print(np.dot(D, X).shape)
                # print(np.outer(D[:, j], X[j, :]).shape)
                E_j = B - np.dot(D, X) + np.outer(D[:, j], X[j, :])
                nonzero_indices = np.nonzero(X[j, :])[0]

                if len(nonzero_indices) == 0:
                    continue

                E_j_restricted = E_j[:, nonzero_indices]
                U, s, Vt = np.linalg.svd(E_j_restricted, full_matrices=False)
                D[:, j] = U[:, 0]
                X[j, nonzero_indices] = s[0] * Vt[0, :]
                
            D = D / np.linalg.norm(D, axis=0)  # Normalize dictionary atoms

        return D, X

    def nmf(self, B):
        """
        Non-negative matrix factorization (NMF) algorithm for dictionary learning.
        """
        nmf = NMF(n_components=self.config.n_atoms, init='random', random_state=0)
        W = nmf.fit_transform(B)
        H = nmf.components_
        return W, H
    
    def dictionary_learning(self,B):
        """
        Dictionary learning using the sklearn DictionaryLearning class.
        """
        self.dict_learning = DictionaryLearning(n_components=self.config.n_atoms, alpha=self.config.alpha_val, max_iter=self.config.num_lasso_iterations, fit_algorithm='cd', transform_algorithm='lasso_lars', positive_code = True, positive_dict = True, n_jobs=1, tol = 1e-4)
        H = self.dict_learning.fit_transform(B)
        # dict_learning.fit(B)

        return self.dict_learning.components_
    
    def denoise_signal(self, noisy_signal):
        """
        Denoise a signal using the learned dictionary D and Lasso for sparse coding.
        """
        noisy_mel_spectrogram = np.array([self.mel_spectrogram(waveform)[0].T for waveform in noisy_signal])
        self.phase_noisy = np.array([self.mel_spectrogram(waveform)[1] for waveform in noisy_signal])
        b, t, f = noisy_mel_spectrogram.shape
        noisy_mel_spectrogram = noisy_mel_spectrogram.reshape(-1, f)
        # print(noisy_signal.shape)
        # sparse_code = self.lasso_sparse_coding(self.D, noisy_signal, self.config.alpha_val)
        # denoised_signal = np.zeros(noisy_signal.shape)

        # sparse_code = np.zeros((self.config.n_atoms, noisy_signal.shape[-1]))
        # for i in range(sparse_code.shape[-1]):
        #         sparse_code[:, i] = self.lasso_sparse_coding(self.D, noisy_signal[:, i], self.config.alpha_val)

        sparse_code = self.dict_learning.transform(noisy_mel_spectrogram)
        denoised_signal = np.dot(self.D.T, sparse_code.T)
        # print(denoised_signal.shape)
        self.denoised_mel_spectrogram = denoised_signal.reshape(b, -1, f)

        return self.denoised_mel_spectrogram
    
    def mel_spectrogram(self, waveform):
        """
        Compute the mel spectrogram of the waveform.
        """
        stft = librosa.stft(waveform, n_fft=self.config.n_fft, hop_length=self.config.hop_size)
        mag, phase = librosa.magphase(stft)
        mel_spectrogram = librosa.feature.melspectrogram(S=mag, sr=self.config.sample_rate, n_mels=self.config.n_mels)
        # print(mel_spectrogram)
        # log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        return mel_spectrogram, phase
    
    def inverse_mel_spectrogram(self, mel_spectrogram, phase):
        """
        Compute the inverse mel spectrogram of the waveform.
        """
        # y_pad = librosa.util.fix_length(np.arange(int(self.config.duration * self.config.sample_rate)), size=n + n_fft // 2)
        # print(mel_spectrogram.shape)
        stft = librosa.feature.inverse.mel_to_stft(mel_spectrogram.T, sr = self.config.sample_rate, n_fft=self.config.n_fft, power=2)
        # print(stft.shape, phase.shape)
        stft = stft * phase
        # print(stft.shape)
        waveform = librosa.istft(stft, hop_length=self.config.hop_size)
        waveform = librosa.util.pad_center(waveform, size=int(self.config.duration * self.config.sample_rate), mode='constant')
        # print(waveform.shape)

        return waveform
    
    def fit(self, clean_speech):
        """
        Learn the dictionary from clean speech signals.
        """
        B = np.array([self.mel_spectrogram(waveform)[0].T for waveform in clean_speech]) # (n_samples, n_frames, n_mels)
        B = B.reshape(-1, B.shape[-1]) # (n_samples * n_frames, n_mels)
        print(f"mel spectrogram shape: {B.shape}")
        # self.D, X = self.k_svd(B) # (n_mels, n_atoms), (n_atoms, n_samples * n_frames)
        self.D = self.dictionary_learning(B)
        print(f"dictionary shape: {self.D.shape}")

    def visualize_spectrogram(self, clean_speech, noisy_speech):
        """
        Visualize the mel spectrogram.
        """
        if len(clean_speech.shape) == 1 and len(noisy_speech.shape) == 1:
            clean_speech = np.expand_dims(clean_speech, axis=0)
            noisy_speech = np.expand_dims(noisy_speech, axis=0)

        fig, axes = plt.subplots(len(clean_speech.shape), 3, figsize=(16, 8))
        for i in range(clean_speech.shape[0]):
            mel_spectrogram_clean, _ = self.mel_spectrogram(clean_speech[i])
            mel_spectrogram_noisy, _ = self.mel_spectrogram(noisy_speech[i])
            mel_spectrogram_denoised = self.denoise_signal([noisy_speech[i]])[0].T

            librosa.display.specshow(librosa.power_to_db(mel_spectrogram_clean, ref=np.max), sr = self.config.sample_rate, n_fft = self.config.n_fft, hop_length = self.config.hop_size, y_axis='mel', x_axis='time', cmap = 'viridis', ax = axes[i,0])
            librosa.display.specshow(librosa.power_to_db(mel_spectrogram_noisy, ref=np.max), sr = self.config.sample_rate, n_fft = self.config.n_fft, hop_length = self.config.hop_size, y_axis='mel', x_axis='time', cmap = 'viridis', ax = axes[i,1])
            librosa.display.specshow(librosa.power_to_db(mel_spectrogram_denoised, ref=np.max), sr = self.config.sample_rate, n_fft = self.config.n_fft, hop_length = self.config.hop_size, y_axis='mel', x_axis='time', cmap = 'viridis', ax = axes[i,2])

        axes[0,0].set_title('Clean signal')
        axes[0,1].set_title('Noisy signal')
        axes[0,2].set_title('Denoised signal')

        plt.tight_layout()
        plt.savefig(f'mel_spectrogram_{self.config.num_ksvd_iterations}_{self.config.noise_type}_{self.config.SNR_dB}.png')
        plt.show()

    def evaluate(self, clean_speech_val):
        """
        Evaluate the denoised signal using PESQ.
        """
        assert len(clean_speech_val) == len(self.denoised_mel_spectrogram), f"Number of clean speech samples ({len(clean_speech_val)}) does not match the number of denoised samples ({len(self.denoised_mel_spectrogram)})."

        # print(self.denoised_mel_spectrogram.shape, self.phase_noisy.shape)
        denoised_signal_inv = np.array([self.inverse_mel_spectrogram(mel_spectrogram, phase) for mel_spectrogram, phase in zip(self.denoised_mel_spectrogram, self.phase_noisy)])
        # print(self.clean_speech.shape, denoised_signal_inv.shape)
        pesq_dict = defaultdict(list)    

        for i in range(len(clean_speech_val)):
            pesq_score_wb = pesq.pesq(fs = self.config.sample_rate, ref = clean_speech_val[i], deg = denoised_signal_inv[i], mode = 'wb')
            pesq_score_nb = pesq.pesq(fs = self.config.sample_rate, ref = clean_speech_val[i], deg = denoised_signal_inv[i], mode = 'nb')
            pesq_dict['wb'].append(pesq_score_wb)
            pesq_dict['nb'].append(pesq_score_nb)

        return pesq_dict


def load_Librispeech_data(dataset_dir, train = True):
    """
    Load the Librispeech dataset for speech enhancement.
    """
    # Load the Librispeech dataset
    dataset_dir = os.path.join(dataset_dir, f"LibriSpeech_{int(config.sample_rate//1000)}kHz_{int(config.duration)}s", ['val', 'train'][train])
    dataset_files = glob(dataset_dir + '/*.wav')
    assert len(dataset_files) > 0, f"No audio files found in the dataset directory. ({dataset_dir})"
    
    audio_data = []

    for file in dataset_files[:10]:
        waveform, sample_rate = librosa.load(file, sr = None, mono = True)
        assert sample_rate == config.sample_rate, f"Sample rate mismatch. Expected {config.sample_rate} but got {sample_rate}."

        audio_data.append(waveform)

    # audio_data = torch.vstack(audio_data, dim=1).numpy()

    return np.asarray(audio_data)


def generate_synthetic_data(clean_speech, dataset_dir):
    """
    Generate synthetic noisy speech signals.
    """
    noise_dir = os.path.join(dataset_dir, 'noise')
    noise_files = glob(os.path.join(noise_dir, f'{config.noise_type}*.wav'))
    assert len(noise_files) > 0, f"No noise files found in the noise directory. ({noise_dir})"

    clean_speech = np.array(clean_speech)
    noise = np.random.choice(noise_files)
    noise, _ = librosa.load(noise, sr = config.sample_rate, mono = True)
    noise = noise[:len(clean_speech[-1])]

    noisy_speech = []
    for clean_signal in clean_speech:
        noise = noise / np.linalg.norm(noise) * np.linalg.norm(clean_signal) * 10**(-config.SNR_dB/20)
        noisy_signal = clean_signal + noise
        noisy_speech.append(noisy_signal)

    return np.asarray(noisy_speech)


if __name__ == "__main__":
    # Example usage
    # clean_signals, noisy_signals = generate_synthetic_data(num_samples, signal_length)
    clean_speech_train = load_Librispeech_data(config.dataset_dir, train = True)
    clean_speech_val = load_Librispeech_data(config.dataset_dir, train = False)
    print("Shape of training samples:", clean_speech_train.shape)
    print("Shape of validation samples:", clean_speech_val.shape)
    noisy_speech_val = generate_synthetic_data(clean_speech_val, config.dataset_dir)
    print("Shape of noisy training samples:", noisy_speech_val.shape)

    speech_enh = SpeechEnhancement(config)
    speech_enh.fit(clean_speech_train)

    # Denoise a sample noisy signal using the learned dictionary
    denoised_signal = speech_enh.denoise_signal(noisy_speech_val)
    print(denoised_signal.shape)
    
    print("Sample clean signal (first sample):", speech_enh.mel_spectrogram(clean_speech_val[0])[0])
    print("Sample noisy signal (first sample):", speech_enh.mel_spectrogram(noisy_speech_val[0])[0])
    print("Denoised signal (first sample):", denoised_signal[0])

    # Evaluate the denoised signal using PESQ
    pesq_dict= speech_enh.evaluate(clean_speech_val)

    print(f"PESQ score (wideband): {np.mean(pesq_dict['wb'])}")
    print(f"PESQ score (narrowband): {np.mean(pesq_dict['nb'])}")

    # Visualize the mel spectrogram
    speech_enh.visualize_spectrogram(clean_speech_val[:2], noisy_speech_val[:2])