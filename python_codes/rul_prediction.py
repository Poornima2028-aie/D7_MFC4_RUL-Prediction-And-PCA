# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    Bidirectional, LSTM, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# DATA LOADING
# ============================================================
columns = ['engine_id', 'cycle'] + \
          [f'op{i}' for i in range(1, 4)] + \
          [f's{i}' for i in range(1, 22)]

train = pd.read_csv(
    r"C:\Users\sowmy\OneDrive\Desktop\mfc4_project\train_FD001.txt",
    delim_whitespace=True, header=None
)
test = pd.read_csv(
    r"C:\Users\sowmy\OneDrive\Desktop\mfc4_project\test_FD001.txt",
    delim_whitespace=True, header=None
)
rul_test = pd.read_csv(
    r"C:\Users\sowmy\OneDrive\Desktop\mfc4_project\RUL_FD001.txt",
    header=None
)

train.columns = columns
test.columns = columns
y_test = rul_test.values.flatten()

# ============================================================
# PIECEWISE RUL LABELING
# ============================================================
RUL_CAP = 125
max_cycle = train.groupby('engine_id')['cycle'].max()

train['RUL'] = train.apply(
    lambda r: max_cycle[r.engine_id] - r.cycle, axis=1
)
train['RUL'] = train['RUL'].clip(upper=RUL_CAP)

# ============================================================
# DATA-DRIVEN SENSOR SELECTION
# ============================================================
sensor_candidates = [f's{i}' for i in range(1, 22)]

# Variance filter
var = train[sensor_candidates].var()
sensor_candidates = var[var > 1e-4].index.tolist()

# Monotonicity
def monotonicity(df, sensor):
    vals = []
    for eid in df.engine_id.unique():
        s = df[df.engine_id == eid][sensor].values
        corr = np.corrcoef(np.arange(len(s)), s)[0, 1]
        vals.append(abs(corr))
    return np.nanmean(vals)

# Prognosability
def prognosability(df, sensor):
    start, end = [], []
    for eid in df.engine_id.unique():
        eng = df[df.engine_id == eid]
        start.append(eng[sensor].iloc[0])
        end.append(eng[sensor].iloc[-1])
    start, end = np.array(start), np.array(end)
    return np.exp(-np.std(end) / (np.mean(np.abs(start - end)) + 1e-6))

scores = {}
for s in sensor_candidates:
    scores[s] = 0.5 * monotonicity(train, s) + 0.5 * prognosability(train, s)

score_df = pd.DataFrame.from_dict(scores, orient='index', columns=['score'])
sensor_cols = score_df.sort_values('score', ascending=False).head(12).index.tolist()

print("Automatically selected sensors:", sensor_cols)

# ============================================================
# NORMALIZATION (FD001 GLOBAL)
# ============================================================
scaler = MinMaxScaler()
train[sensor_cols] = scaler.fit_transform(train[sensor_cols])
test[sensor_cols] = scaler.transform(test[sensor_cols])

# ============================================================
# SEQUENCE CREATION
# ============================================================
WINDOW = 30

def create_train_sequences(data):
    X, y = [], []
    for eid in data.engine_id.unique():
        df = data[data.engine_id == eid]
        vals = df[sensor_cols].values
        rul = df['RUL'].values
        for i in range(len(vals) - WINDOW):
            X.append(vals[i:i+WINDOW])
            y.append(rul[i+WINDOW])
    return np.array(X), np.array(y)

def create_test_sequences(data):
    X = []
    for eid in data.engine_id.unique():
        df = data[data.engine_id == eid]
        X.append(df[sensor_cols].values[-WINDOW:])
    return np.array(X)

X_train, y_train = create_train_sequences(train)
X_test = create_test_sequences(test)

# ============================================================
# ATTENTION LAYER
# ============================================================
class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )

    def call(self, x):
        e = tf.matmul(x, self.W)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(a * x, axis=1)

# ============================================================
# MODEL
# ============================================================
inp = Input(shape=(WINDOW, len(sensor_cols)))

x = Conv1D(32, 3, activation='relu')(inp)
x = MaxPooling1D(2)(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Attention()(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(1)(x)

model = Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='mse'
)

model.summary()

# ============================================================
# TRAINING
# ============================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# EVALUATION
# ============================================================
y_pred = model.predict(X_test).flatten()
y_pred = np.clip(y_pred, 0, RUL_CAP)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

TOL = 40
acc = np.mean(np.abs(y_pred - y_test) <= TOL) * 100

print("\n===== MODEL PERFORMANCE =====")
print(f"RMSE                     : {rmse:.3f}")
print(f"MAE                      : {mae:.3f}")
print(f"R² Score                 : {r2:.3f}")
print(f"Accuracy (±{TOL} cycles) : {acc:.2f}%")

# ============================================================
# VISUALIZATIONS
# ============================================================

# 1. Training vs Validation Loss
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# 2. Predicted vs True RUL
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([0, RUL_CAP], [0, RUL_CAP], 'r--')
plt.xlabel('True RUL')
plt.ylabel('Predicted RUL')
plt.title('Predicted vs True RUL')
plt.grid(True)
plt.show()

# 3. Error Distribution
errors = y_pred - y_test
plt.figure(figsize=(6,4))
plt.hist(errors, bins=30)
plt.xlabel('Prediction Error (cycles)')
plt.ylabel('Frequency')
plt.title('Distribution of RUL Prediction Errors')
plt.grid(True)
plt.show()

# 4. Error vs True RUL
plt.figure(figsize=(6,4))
plt.scatter(y_test, errors, alpha=0.7)
plt.axhline(0, linestyle='--')
plt.xlabel('True RUL')
plt.ylabel('Prediction Error')
plt.title('Prediction Error vs True RUL')
plt.grid(True)
plt.show()
