# sEMG Gesture Recognition: From Reservoir Computing to TCN

This repository implements and compares two advanced architectures for gesture classification using surface Electromyography (sEMG) data from the NinaPro DB1 dataset. The project tracks the evolution of a model from a Deep Echo State Network (ESN) to a high-performing Temporal Convolutional Network (TCN).

## Experimental performance summary

Deep-Rich ESN		71.42%
Temporal ConvNet (TCN)	74.89%
## Architectures

**1. Deep-Rich Echo State Network (ESN)**
A reservoir computing approach utilizing:
- Deep Architecture: Two stacked reservoir layers to extract hierarchical temporal dynamics.
- Rich Feature Extraction: Manual extraction of MAV (Mean Absolute Value), Max, and Min within the temporal window to bolster the reservoir's "memory."
- Regularization: Optimized using a Ridge Classifier ($\alpha = 0.01$) to handle high-dimensional feature spaces (3,000+ features).
**2. Temporal Convolutional Network (TCN)**
A deep learning approach designed to replace manual feature engineering with learned temporal filters:
- Dilated Convolutions: Used to expand the receptive field exponentially without losing resolution or increasing parameter count excessively.
- Causal Padding: Ensures the model only utilizes past and current samples, making it viable for real-time applications.
- Optimization: Trained using Adam with a ReduceLROnPlateau scheduler and Gradient Norm Clipping to stabilize training across 53 gesture classes.

## Key Results
** Confusion Matrix Analysis**

The final TCN model achieves superior separation in the majority of the 53 classes. Analysis of the confusion matrix reveals that remaining errors are primarily concentrated in Gestures 9 and 11 (Thumb adduction and Thumb flexion).

## Installation & Usage
**Prerequisites**

    Python 3.8+

    PyTorch

    Scikit-learn

    NumPy / Pandas

    Matplotlib / Seaborn

**Setup**
```bash

git clone https://github.com/your-username/nina-pro-emg-tcn.git
cd nina-pro-emg-tcn
pip install -r requirements.txt
```
**Model parameters**

For the ESN:

n_layers = 2
n_res_per_layer = 500
n_inputs = 10
alpha_l1 = 0.2
alpha_l2 = 0.05
input_scale = 0.1

For the TCN:

num_channels=[64, 64, 64]
num_inputs=10
kernel_size=3
dropout=0.2

**Downloading data**

Download s1.zip from the Ninapro DB1 dataset, extract the S1_A1_E1.mat file and place it inside the data folder of this repository
Link to Ninapro: https://ninapro.hevs.ch/

**Reproducing Results**

To train the ESN model:
```bash

python Deep_ESN.py
```

To train the TCN model:
```bash

# Ensure data is normalized and windowed (20 samples/200ms)
python TCN.py --epochs 150 --lr 0.01

To load the models using pre-trained weights run the cells at demo.ipynb
```
## Learning Journey

This project documents a transition from stochastic dynamics (Reservoirs) to structured feature learning (Convolutions).
- Initial ESN baselines started at ~11%.
- Temporal integration and leakage optimization pushed results to 54%.
- Deep-Rich feature engineering achieved a breakthrough at 72%.
- The final TCN architecture bypassed the reservoir "noise" to reach the state-of-the-art threshold of ~77%.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Data provided by the NinaPro (Non-Invasive Adaptive Hand Prosthetics) project.
Inspired by research in Reservoir Computing and Dilated Convolutions for bio-signal processing.
