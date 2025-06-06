import sys
import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
import lab1_ndpcm_library  # Ensure this library is available

### Configuration parameters
n_bits = 8 if len(sys.argv) < 2 else int(sys.argv[1])  # Quantization bits
n_values = [10, 50, 100, 200]  # Different iteration counts to test
h_depth = 3  # Fixed history depth

### Prepare figure
plt.figure(figsize=(12, 8))

for idx, n in enumerate(n_values):
    # Generate sine wave data (0 ~ 2^n_bits)
    x = np.linspace(0, 2 * pi, n)
    f_original = np.sin(x)
    f = (f_original + 1) * (2 ** (n_bits - 1))

    # Initialize codec
    tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
    rx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)

    # Main processing loop
    for k in range(1, n - 1):
        # Transmitter processing
        lab1_ndpcm_library.prepare_params_for_prediction(tx_data, k)
        y_hat = lab1_ndpcm_library.predict(tx_data, k)
        eq = lab1_ndpcm_library.calculate_error(tx_data, k, f[k])
        y_rec = lab1_ndpcm_library.reconstruct(tx_data, k)
        
        # Receiver processing (simulate transmission)
        rx_data.eq[k] = tx_data.eq[k]
        lab1_ndpcm_library.prepare_params_for_prediction(rx_data, k)
        y_hat_rx = lab1_ndpcm_library.predict(rx_data, k)
        y_rec_rx = lab1_ndpcm_library.reconstruct(rx_data, k)

    # Plot results comparison
    plt.subplot(2, 2, idx + 1)
    plt.plot(f, 'b-', linewidth=2, label="Original signal")
    plt.plot(rx_data.y_recreated, 'r--', label="Reconstructed signal")
    
    # Calculate and display error metrics
    mse = np.mean((f - rx_data.y_recreated) ** 2)
    max_error = np.max(np.abs(f - rx_data.y_recreated))
    
    plt.title(f"n={n}\nMSE={mse:.2f}, MaxErr={max_error:.1f}")
    plt.xlabel("Sample points")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.savefig(f"n_comparison_bits_{n_bits}.png")
plt.show()

# Additional plot for quantization error comparison
plt.figure(figsize=(10, 6))
for n in n_values:
    # Regenerate data (simplified)
    x = np.linspace(0, 2 * pi, n)
    f = (np.sin(x) + 1) * (2 ** (n_bits - 1))
    
    # Run codec process (simplified)
    tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
    for k in range(1, n - 1):
        lab1_ndpcm_library.prepare_params_for_prediction(tx_data, k)
        lab1_ndpcm_library.calculate_error(tx_data, k, f[k])
    
    # Plot quantization error
    plt.plot(np.abs(tx_data.eq), label=f"n={n}")

plt.title("Quantization Error Magnitude Comparison")
plt.xlabel("Sample points")
plt.ylabel("|Quantization error|")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"quant_error_comparison_bits_{n_bits}.png")
plt.show()
