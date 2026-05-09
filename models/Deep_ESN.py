# Loading .mat files and visualization
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 1. Load the data
base_path = Path(__file__).parent.parent
data_path = base_path / "data" / "S1_A1_E1.mat"

data = sio.loadmat(data_path)

# 2. Extract key variables
emg = data['emg']            # The 10-channel EMG signal
labels = data['restimulus']  # The gesture IDs (0 to 52)
reps = data['repetition']    # Which repetition (1-10) it belongs to

# Nrmalization is critical in an Echo State Network (ESN). ESNs use non-linear activation functions (usually tanh) in the reservoir. If your input EMG values are too large, they will "saturate" the neurons (pushing them all to 1 or -1), and your reservoir will lose its ability to distinguish between different signals.
# For NinaPro DB1, the most common and effective method is Min-Max Scaling to the range [0,1].

from sklearn.preprocessing import MinMaxScaler

# 1. Identify your Training and Testing splits based on repetitions
# Example: Train on Reps 1, 3, 4, 6, 8, 9; Test on 2, 5, 7, 10
train_mask = np.isin(reps, [1, 3, 4, 6, 8, 9]).flatten()
test_mask = np.isin(reps, [2, 5, 7, 10]).flatten()

emg_train = emg[train_mask]
emg_test = emg[test_mask]

# 2. Initialize and Fit the Scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit only on training data
scaler.fit(emg_train)

# 3. Transform both sets
emg_train_norm = scaler.transform(emg_train)
emg_test_norm = scaler.transform(emg_test)

print(f"Normalized Range: {emg_train_norm.min()} to {emg_train_norm.max()}")

# Windowing: An ESN processes a sequence. If we just give it one sample at a time (at 100Hz, that's only 10ms of data), it's like trying to understand a word by looking at a single pixel of one letter. We need Windows to provide enough context for the reservoir to "vibrate" in a meaningful way.
#The Plan:

# Window Length (W): 20 samples (200ms). This is long enough to capture the "intent" of a muscle contraction.

# Step Size (S): 10 samples (100ms). This creates a 50% overlap, which gives us more training data and smoother transitions between gestures.
def create_windows(data, labels, window_size=20, step_size=10):
    n_samples = data.shape[0]
    n_channels = data.shape[1]
    
    windows = []
    window_labels = []
    
    for i in range(0, n_samples - window_size, step_size):
        # Extract the window
        window = data[i : i + window_size, :]
        
        # For the label, we usually take the most frequent label in that window (the Mode)
        # or simply the label at the very end of the window.
        label = labels[i + window_size - 1] 
        
        windows.append(window)
        window_labels.append(label)
        
    return np.array(windows), np.array(window_labels)

# Apply it to your data
X_train, y_train = create_windows(emg_train_norm, labels[train_mask])
X_test, y_test = create_windows(emg_test_norm, labels[test_mask])

print(f"Training shape: {X_train.shape}") 
# Expected: (Number of Windows, 20 samples, 10 channels)
# Data is now structured in a 3D tensor [Windows,Time,Features]


# Hybrid feature extraction

# Right now, you are feeding the ESN raw (though normalized) EMG data. In biological signal processing, we can help the reservoir by giving it "pre-digested" mathematical features alongside the raw signal.

#The Strategy: For each 20-sample window, calculate the Mean Absolute Value (MAV) and Root Mean Square (RMS). These are the industry standards for muscle effort.
# Updated Windowing with Features:

def create_hybrid_windows(data, labels, window_size=20, step_size=10):
    n_samples, n_channels = data.shape
    windows = []
    window_labels = []
    features = [] # To store MAV/RMS
    
    for i in range(0, n_samples - window_size, step_size):
        window = data[i : i + window_size, :]
        
        # 1. Standard window for the ESN
        windows.append(window)
        
        # 2. Calculate MAV and RMS for this window (10 values each)
        mav = np.mean(np.abs(window), axis=0)
        rms = np.sqrt(np.mean(window**2, axis=0))
        
        # Combine them into a 20-feature vector
        features.append(np.concatenate([mav, rms]))
        
        window_labels.append(labels[i + window_size - 1])
        
    return np.array(windows), np.array(features), np.array(window_labels)

# Re-run windowing
X_train_raw, X_train_feat, y_train = create_hybrid_windows(emg_train_norm, labels[train_mask])
X_test_raw, X_test_feat, y_test = create_hybrid_windows(emg_test_norm, labels[test_mask])

# Hyperparameters for DeepESN
n_layers = 2
n_res_per_layer = 500
n_inputs = 10
alpha_l1 = 0.2      # We can use different leak rates per layer!
alpha_l2 = 0.05
input_scale = 0.1

# 1. Initialize Weights for each layer
# Win only connects to the first layer
Win = (np.random.rand(n_res_per_layer, n_inputs) - 0.5) * input_scale

# Internal reservoir weights and inter-layer weights
Wres_list = []
Winter_list = [] # Connections between Layer 1 -> Layer 2

for i in range(n_layers):
    # Internal Reservoir (Sparse)
    mask = np.random.rand(n_res_per_layer, n_res_per_layer) < 0.1
    W = (np.random.rand(n_res_per_layer, n_res_per_layer) - 0.5) * mask
    
    # Scale Spectral Radius for stability
    sr = np.max(np.abs(np.linalg.eigvals(W)))
    W *= (0.95 / sr)
    Wres_list.append(W)
    
    # Connection from Layer (i) to Layer (i+1)
    if i < n_layers - 1:
        W_inter = (np.random.rand(n_res_per_layer, n_res_per_layer) - 0.5) * 0.1
        Winter_list.append(W_inter)

from sklearn.linear_model import RidgeClassifier
# Deep state update
# Each layer's state at time t depends on its own previous state and the current state of the layer below it.
def get_rich_deep_states(X_data, n_layers, n_res, Win, Wres_list, Winter_list, alpha_l1, alpha_l2):
    n_windows, window_len, _ = X_data.shape
    
    # CRITICAL: We use n_res (which is 500) here
    all_features = np.zeros((n_windows, n_layers * n_res * 3))
    
    for i in range(n_windows):
        # Initialize each layer with the correct layer size (500)
        states = [np.zeros(n_res) for _ in range(n_layers)]
        window_data = X_data[i]
        
        # Track history for all layers: Shape (window_len, n_layers, n_res)
        history = np.zeros((window_len, n_layers, n_res))
        
        for t in range(window_len):
            u = window_data[t]
            
            # Layer 1 Update
            l1_input = np.dot(Win, u) + np.dot(Wres_list[0], states[0])
            states[0] = (1 - alpha_l1) * states[0] + alpha_l1 * np.tanh(l1_input)
            history[t, 0, :] = states[0]
            
            # Subsequent Layers Update
            for L in range(1, n_layers):
                l_input = np.dot(Winter_list[L-1], states[L-1]) + np.dot(Wres_list[L], states[L])
                states[L] = (1 - alpha_l2) * states[L] + alpha_l2 * np.tanh(l_input)
                history[t, L, :] = states[L]
        
        # Flatten the history for statistics
        layer_stats = []
        for L in range(n_layers):
            layer_history = history[:, L, :] # Shape (20, 500)
            layer_stats.append(np.mean(layer_history, axis=0))
            layer_stats.append(np.max(layer_history, axis=0))
            layer_stats.append(np.min(layer_history, axis=0))
            
        all_features[i, :] = np.concatenate(layer_stats)
        
    return all_features

# Execute
X_train_ultimate = get_rich_deep_states(X_train_raw, 2, 500, Win, Wres_list, Winter_list, alpha_l1, alpha_l2)
X_test_ultimate = get_rich_deep_states(X_test_raw, 2, 500, Win, Wres_list, Winter_list, alpha_l1, alpha_l2)

# Append MAV/RMS (optional but recommended)
X_train_final = np.hstack([X_train_ultimate, X_train_feat])
X_test_final = np.hstack([X_test_ultimate, X_test_feat])

classifier = RidgeClassifier(alpha=0.01, class_weight='balanced') # Changing the value of alpha can prevent overfitting on training data
classifier.fit(X_train_final, y_train.ravel())
print(f"Final Deep-Rich Accuracy: {classifier.score(X_test_final, y_test.ravel())*100:.2f}%")

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = classifier.predict(X_test_final)
cm = confusion_matrix(y_test, y_pred)
# Normalization (to see percentages instead of raw window counts)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized,annot=False, cmap='Blues', fmt='.2f')
plt.title('Gesture Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

import joblib

# Save the trained Ridge Classifier
joblib.dump(classifier, base_path / "weights" / "esn_readout_model.pkl")