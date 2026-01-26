 ============================================================
# TURBOFAN ENGINE RUL PREDICTION (C-MAPSS)
# SEPARATE TRAINING FOR FD001, FD002, FD003, FD004
# CNN + BiLSTM + Attention
# ============================================================

# ======================
# IMPORTS
# ======================
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

# ======================
# GLOBAL SETTINGS
# ======================
BASE_PATH = r"C:\Users\sowmy\OneDrive\Desktop\mfc4_project\cmappss_dataset"
WINDOW = 30
TOL = 40

DATASETS = {
    "FD001": 125,
    "FD002": 150,
    "FD003": 125,
    "FD004": 150
}

columns = ['engine_id', 'cycle'] + \
          [f'op{i}' for i in range(1, 4)] + \
          [f's{i}' for i in range(1, 22)]

# ======================
# DATA LOADER
# ======================
def load_dataset(dataset):
    train = pd.read_csv(
        f"{BASE_PATH}/train_{dataset}.txt",
        delim_whitespace=True, header=None
    )
    test = pd.read_csv(
        f"{BASE_PATH}/test_{dataset}.txt",
        delim_whitespace=True, header=None
    )
    rul = pd.read_csv(
        f"{BASE_PATH}/RUL_{dataset}.txt",
        header=None
    )

    train.columns = columns
    test.columns = columns

    return train, test, rul.values.flatten()

# ======================
# SENSOR SELECTION
# ======================
def monotonicity(df, sensor):
    vals = []
    for eid in df.engine_id.unique():
        s = df[df.engine_id == eid][sensor].values
        corr = np.corrcoef(np.arange(len(s)), s)[0, 1]
        vals.append(abs(corr))
    return np.nanmean(vals)

def prognosability(df, sensor):
    start, end = [], []
    for eid in df.engine_id.unique():
        eng = df[df.engine_id == eid]
        start.append(eng[sensor].iloc[0])
        end.append(eng[sensor].iloc[-1])
    start, end = np.array(start), np.array(end)
    return np.exp(-np.std(end) / (np.mean(np.abs(start - end)) + 1e-6))

def select_sensors(train):
    sensors = [f's{i}' for i in range(1, 22)]

    var = train[sensors].var()
    sensors = var[var > 1e-4].index.tolist()

    scores = {}
    for s in sensors:
        scores[s] = 0.5 * monotonicity(train, s) + \
                    0.5 * prognosability(train, s)

    score_df = pd.DataFrame.from_dict(
        scores, orient='index', columns=['score']
    )
    return score_df.sort_values('score', ascending=False)\
                   .head(12).index.tolist()

# ======================
# SEQUENCE CREATION
# ======================
def create_train_sequences(df, sensors, RUL_CAP):
    X, y = [], []

    max_cycle = df.groupby('engine_id')['cycle'].max()
    df['RUL'] = df.apply(
        lambda r: max_cycle[r.engine_id] - r.cycle, axis=1
    )
    df['RUL'] = df['RUL'].clip(upper=RUL_CAP)

    for eid in df.engine_id.unique():
        d = df[df.engine_id == eid]
        vals = d[sensors].values
        rul = d['RUL'].values

        for i in range(len(vals) - WINDOW):
            X.append(vals[i:i + WINDOW])
            y.append(rul[i + WINDOW])

    return np.array(X), np.array(y)

def create_test_sequences(df, sensors):
    X = []

    for eid in df.engine_id.unique():
        d = df[df.engine_id == eid]
        seq = d[sensors].values

        if len(seq) >= WINDOW:
            X.append(seq[-WINDOW:])
        else:
            pad_len = WINDOW - len(seq)
            pad = np.repeat(seq[0:1], pad_len, axis=0)
            X.append(np.vstack((pad, seq)))

    return np.array(X)

# ======================
# ATTENTION LAYER
# ======================
class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='normal',
            trainable=True
        )

    def call(self, x):
        e = tf.matmul(x, self.W)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(a * x, axis=1)

# ======================
# MODEL
# ======================
def build_model(n_features):
    inp = Input(shape=(WINDOW, n_features))

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
    return model

# ======================
# MAIN TRAINING LOOP
# ======================
results = {}

for dataset, RUL_CAP in DATASETS.items():
    print(f"\n========== TRAINING {dataset} ==========")

    train, test, y_test = load_dataset(dataset)

    sensor_cols = select_sensors(train)
    print("Selected sensors:", sensor_cols)

    scaler = MinMaxScaler()
    train[sensor_cols] = scaler.fit_transform(train[sensor_cols])
    test[sensor_cols] = scaler.transform(test[sensor_cols])

    X_train, y_train = create_train_sequences(train, sensor_cols, RUL_CAP)
    X_test = create_test_sequences(test, sensor_cols)

    model = build_model(len(sensor_cols))
    model.summary()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test).flatten()
    y_pred = np.clip(y_pred, 0, RUL_CAP)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    acc = np.mean(np.abs(y_pred - y_test) <= TOL) * 100

    results[dataset] = (rmse, mae, r2, acc)

    print(f"\n{dataset} RESULTS")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"R²   : {r2:.3f}")
    print(f"Accuracy (±{TOL}) : {acc:.2f}%")

# ======================
#h FINAL SUMMARY
# ======================
print("\n========== FINAL SUMMARY ==========")
for d, r in results.items():
    print(
        f"{d} -> RMSE={r[0]:.2f}, MAE={r[1]:.2f}, "
        f"R²={r[2]:.3f}, Acc={r[3]:.2f}%"
    )
