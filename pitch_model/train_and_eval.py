"""
Train and evaluate pitch velocity prediction.
Generates synthetic biomechanics data mimicking the OpenBiomechanics Project.
Prints metrics in key_value format for the scoring pipeline.
"""

import time
import numpy as np
from model import build_model, extract_features

np.random.seed(42)

# --- Synthetic biomechanics data ---
# 10 features: shoulder rotation, hip angle, trunk tilt, stride length, etc.
N_SAMPLES = 600
N_FEATURES = 10

X = np.random.randn(N_SAMPLES, N_FEATURES)

# Ground truth: pitch speed is a nonlinear function of biomechanics
weights = np.array([3.5, 2.8, 1.9, 1.5, 1.0, 0.7, 0.4, 0.2, 0.1, 0.05])
y = 88.0 + X @ weights + 0.3 * X[:, 0] * X[:, 1] + np.random.randn(N_SAMPLES) * 2.5

# Train/test split (80/20)
split = int(0.8 * N_SAMPLES)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model and extract features
start = time.time()
model = build_model()
X_train_feat = extract_features(X_train)
model.fit(X_train_feat, y_train)
train_time = time.time() - start

# Evaluate on test set
X_test_feat = extract_features(X_test)
preds = model.predict(X_test_feat)

rmse = np.sqrt(np.mean((preds - y_test) ** 2))
mae = np.mean(np.abs(preds - y_test))
ss_res = np.sum((y_test - preds) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r_squared = 1.0 - ss_res / ss_tot

# Print metrics in key_value format
print("---")
print(f"rmse_mph: {rmse:.6f}")
print(f"mae_mph: {mae:.6f}")
print(f"r_squared: {r_squared:.6f}")
print(f"train_time_s: {train_time:.1f}")
