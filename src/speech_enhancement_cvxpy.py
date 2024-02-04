import numpy as np
import cvxpy as cp

def generate_synthetic_data(num_samples, signal_length, noise_level=0.1):
    """
    Generate synthetic clean and noisy speech signals.
    """
    clean_signals = np.random.rand(num_samples, signal_length)
    noise = noise_level * np.random.randn(num_samples, signal_length)
    noisy_signals = clean_signals + noise
    return clean_signals, noisy_signals

def learn_dictionary_via_optimization(clean_signals, n_atoms):
    """
    Learn a dictionary from clean speech signals using a simple optimization model.
    """
    m, n = clean_signals.shape
    D = cp.Variable((m, n_atoms))
    X = cp.Variable((n_atoms, n))
    
    # Objective: Minimize ||D*X - clean_signals||_F^2
    objective = cp.Minimize(cp.norm(D @ X - clean_signals, 'fro'))
    constraints = [cp.norm(D, 'fro') <= 1]  # Example constraint to avoid trivial solution
    prob = cp.Problem(objective, constraints)
    
    prob.solve()
    return D.value

def denoise_signal_via_optimization(noisy_signal, dictionary, lambda_val=0.1):
    """
    Denoise a speech signal using the learned dictionary and sparse coding via optimization.
    """
    n_atoms = dictionary.shape[1]
    x = cp.Variable(n_atoms)
    
    # Objective: Minimize ||dictionary*x - noisy_signal||_2^2 + lambda_val*cp.norm(x, 1)
    objective = cp.Minimize(cp.norm(dictionary @ x - noisy_signal, 2)**2 + lambda_val*cp.norm(x, 1))
    prob = cp.Problem(objective)
    
    prob.solve()
    return dictionary @ x.value

def speech_enhancement_pipeline(num_samples=10, signal_length=256, n_atoms=50, lambda_val=0.1):
    """
    Complete pipeline for learning a dictionary and denoising speech signals.
    """
    clean_signals, noisy_signals = generate_synthetic_data(num_samples, signal_length)
    
    # Learn dictionary from clean signals
    learned_dictionary = learn_dictionary_via_optimization(clean_signals.T, n_atoms)
    
    # Denoise a sample noisy signal using the learned dictionary
    sample_noisy_signal = noisy_signals[0]  # Take the first noisy signal for demonstration
    denoised_signal = denoise_signal_via_optimization(sample_noisy_signal, learned_dictionary, lambda_val)
    
    return sample_noisy_signal, denoised_signal

# Example usage
if __name__ == "__main__":
    sample_noisy_signal, denoised_signal = speech_enhancement_pipeline()
    
    print("Sample noisy signal (first 10 samples):", sample_noisy_signal[:10])
    print("Denoised signal (first 10 samples):", denoised_signal[:10])
