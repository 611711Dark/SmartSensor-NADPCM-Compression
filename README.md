# NADPCM Compression Algorithm Experiment

## Index
- [中文 README](README_cn.md)

## English README

### Overview
This repository contains Python implementations and experiments for Nonlinear Adaptive Pulse Coded Modulation-Based Compression (NADPCMC) as part of a smart sensors laboratory. The project explores adaptive compression techniques for wireless sensor networks, evaluating performance metrics like distortion rate, compression ratio, and reconstruction error under various configurations.

### Key Features
- **Core Algorithms**: Implementation of NADPCM encoder/decoder
- **Experiments**:
  - Bit-depth variation (8-16 bits)
  - Signal frequency variation (1-20Hz)
  - Three signal types: slow/fast/very fast sine waves
- **Performance Metrics**:
  - Distortion percentage
  - Compression ratio
  - Average/max reconstruction error
- **Visualization**: Automatic generation of comparative result plots

### File Structure
```
nadpcm_experiment.py            # Main experiment with bit-depth variation
nadpcm_variable_freq_experiment.py # Frequency sweep experiment
lab1-template.py                # Basic implementation template
lab1_ndpcm_library.py           # Core NADPCM algorithms
lab1_library.py                 # Quantization functions
```

### Dependencies
- Python 3.7+
- NumPy
- Matplotlib

### How to Run
1. Execute main experiment:
```bash
python nadpcm_experiment.py
```
2. Run frequency experiment:
```bash
python nadpcm_variable_freq_experiment.py
```

### Sample Results
```
============================================================
DISTORTION RATE PER SIGNAL AND BIT DEPTH
============================================================
Signal/Bits         8-bit          12-bit         13-bit         14-bit         16-bit
-----------------------------------------------------------------------------------------------
slow_sine           9.01e+18%      3.07%          3.71%          3.71%          3.71%
fast_sine           7.68e+20%      4.23%          3.84%          3.84%          3.84%
very_fast_sine      1.50e+06%      4.95e+43%      7.59e+44%      1.98e+48%      2.06%

Frequency  Distortion (%)  Avg Error
----------------------------------------
1.00       1.96            1.9243 × 10²
1.50       1.90            2.1111 × 10²
2.00       2.10            2.2943 × 10²
2.50       2.06            2.5025 × 10²
3.00       2.24            2.8051 × 10²
3.50       2.9626 × 10²²   3.1628 × 10²⁴

```

**Critical Finding**: Distortion exceeds 20% at 14-bit encoding for 10Hz signals

### References
- Nonlinear Adaptive Pulse Coded Modulation theory
- Wireless Sensor Network compression techniques
- Adaptive estimation algorithms


