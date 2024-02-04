import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import Lasso

def generate_synthetic_data(num_samples, signal_length, noise_level=0.1):
    """
    Generate synthetic clean and noisy speech signals for demonstration.
    """
    # Generate clean signals
    clean_signals = np.random.rand(num_samples, signal_length)
    
    # Add noise to generate noisy signals
    noise = noise_level * np.random.randn(num_samples, signal_length)
    noisy_signals = clean_signals + noise
    
    return clean_signals, noisy_signals

def learn_dictionary(clean_signals, n_components):
    """
    Learn a dictionary from clean speech signals using Dictionary Learning.
    """
    dict_learner = DictionaryLearning(n_components=n_components, 
                                      transform_algorithm='lasso_lars',
                                      fit_algorithm='lars',
                                      random_state=42)
    dict_learner.fit(clean_signals)
    return dict_learner.components_

def denoise_signal(noisy_signal, dictionary, lambda_val):
    """
    Denoise a speech signal using the learned dictionary and sparse coding.
    """
    lasso = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=1000)
    lasso.fit(dictionary.T, noisy_signal)  # Transpose dictionary to match dimensions
    sparse_code = lasso.coef_
    denoised_signal = np.dot(dictionary.T, sparse_code)
    return denoised_signal

def speech_enhancement_pipeline(num_samples=50, signal_length=256, n_components=100, lambda_val=0.1):
    """
    Complete pipeline for learning a dictionary and denoising speech signals.
    """
    # Step 1: Generate synthetic clean and noisy speech signals
    clean_signals, noisy_signals = generate_synthetic_data(num_samples, signal_length)
    
    # Step 2: Learn dictionary from clean signals
    learned_dictionary = learn_dictionary(clean_signals, n_components)
    
    # Step 3: Denoise a sample noisy signal using the learned dictionary
    sample_noisy_signal = noisy_signals[0]  # Take the first noisy signal for demonstration
    denoised_signal = denoise_signal(sample_noisy_signal, learned_dictionary, lambda_val)
    
    return sample_noisy_signal, denoised_signal

# Example usage
if __name__ == "__main__":
    sample_noisy_signal, denoised_signal = speech_enhancement_pipeline()
    
    print("Sample noisy signal:", sample_noisy_signal[:10])  # Display first 10 samples
    print("Denoised signal:", denoised_signal[:10])  # Display first 10 samples
