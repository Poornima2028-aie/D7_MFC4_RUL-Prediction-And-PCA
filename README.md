# 🚀 D7_MFC4_RUL Prediction

## ✈️ Attention-Based Remaining Useful Life (RUL) Prediction for Aircraft Turbofan Engines

---

# 📌 Project Title

**Attention-Based Remaining Useful Life Prediction for Aircraft Turbofan Engines using Deep Learning**

---

# 👥 Team Members

- **Poornima P** – cb.sc.u4aie24343  
- **Ch. Sarvani Sruthi** – cb.sc.u4aie24311  
- **Shri Manasa** – cb.sc.u4aie24356  
- **Sowmya A** – cb.sc.u4aie24357  

---

# 🎯 Objective

The objective of this project is to predict the **Remaining Useful Life (RUL)** of aircraft turbofan engines using **multivariate time-series sensor data**.

The aim is to enable **predictive maintenance** by estimating how long an engine can operate before failure. This helps to:

- Improve aircraft safety  
- Reduce unexpected downtime  
- Minimize maintenance costs  
- Optimize maintenance scheduling  

By analyzing degradation patterns in sensor data, the model predicts the **number of cycles remaining before engine failure**.

---

# 💡 Motivation / Why the Project is Interesting

Aircraft engine maintenance is highly critical and expensive. Traditional maintenance strategies either replace components **too early** or **too late**, which leads to increased operational costs or safety risks.

This project is interesting because it:

- Uses **real-world NASA engine degradation data**
- Applies **deep learning models to multivariate time-series sensor data**
- Learns degradation patterns automatically from sensor measurements
- Uses an **attention mechanism** to identify the most important degradation periods

By learning degradation behaviour directly from the data, the system can predict engine failure in advance and support **predictive maintenance strategies**.

---

# 🛠️ Methodology

---

# 📂 Dataset

The dataset used in this project is the **NASA C-MAPSS Turbofan Engine Dataset**.

The dataset contains **multivariate time-series sensor measurements** collected from aircraft engines operating under different conditions until failure.

Experiments are conducted on the following subsets:

- FD002  
- FD003  
- FD004  

Each dataset contains:

- Engine ID  
- Cycle number  
- Operational settings  
- Multiple sensor measurements  

Each row represents **one operating cycle of a specific engine**.

As the number of cycles increases, the engine gradually degrades until failure.

---

# 🔄 Data Preprocessing

## Normalization

Sensor values are normalized to ensure that all features lie within a similar numerical range.

The normalization formula used is:

$$
X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

This ensures stable training and prevents large-value features from dominating the learning process.

---
---

# 📡 Sensor Selection

Before applying the sliding window technique, an important preprocessing step is **sensor selection**.

The C-MAPSS turbofan dataset contains many sensors, but **not all sensors contribute useful degradation information**. Some sensors fluctuate randomly and do not reflect the health of the engine.

Therefore, two important metrics are used to evaluate the usefulness of sensors:

- **Monotonicity**
- **Prognosability**

Sensors that score high in these metrics are selected for training the deep learning model.

---

# 1️⃣ Monotonicity

Monotonicity measures **whether a sensor consistently increases or decreases over time**.

If a sensor steadily increases or decreases across engine cycles, it indicates that the sensor reflects **engine degradation behaviour**.

This relationship is measured using the **Pearson correlation coefficient** between **time (cycle number)** and **sensor values**.

The correlation value ranges between:

| Correlation Value | Meaning |
|---|---|
| +1 | Perfect increasing trend |
| -1 | Perfect decreasing trend |
| 0 | No relationship |

In degradation analysis, the **absolute value of correlation** is used:

\[
Monotonicity = |corr(time, sensor)|
\]

Both increasing and decreasing trends are useful because they indicate **consistent degradation patterns**.

---

## Example: Good Monotonic Sensor

Consider **Sensor S7** from one engine.

| Cycle | Sensor S7 |
|---|---|
|1|50|
|2|52|
|3|55|
|4|58|
|5|61|

Time values:

\[
[1,2,3,4,5]
\]

Sensor values:

\[
[50,52,55,58,61]
\]

Computing the Pearson correlation between time and sensor value gives:

\[
corr = 0.99
\]

Absolute value:

\[
|0.99| = 0.99
\]

### Interpretation

- The sensor increases steadily with time  
- Correlation is close to **1**  
- The sensor strongly reflects degradation  

Therefore **Sensor S7 has high monotonicity**.

---

## Example: Poor Monotonic Sensor

Consider **Sensor S3**.

| Cycle | Sensor S3 |
|---|---|
|1|50|
|2|53|
|3|49|
|4|55|
|5|52|

Time:

\[
[1,2,3,4,5]
\]

Sensor values:

\[
[50,53,49,55,52]
\]

Correlation:

\[
corr = 0.21
\]

Absolute value:

\[
|0.21| = 0.21
\]

### Interpretation

- The sensor fluctuates randomly  
- No clear degradation trend  
- Monotonicity is low  

Therefore this sensor is **not useful for degradation modeling**.

---

## Averaging Across Engines

The turbofan dataset contains **multiple engines**, each with its own degradation trajectory.

Therefore monotonicity must be computed **for every engine separately**.

The final monotonicity score is calculated as the **average of absolute correlations across engines**:

\[
Monotonicity = \frac{1}{N}\sum_{i=1}^{N} |corr_i|
\]

Where:

- \(N\) = number of engines  
- \(corr_i\) = correlation for engine \(i\)

---

# 2️⃣ Prognosability Metric

Prognosability measures **whether different engines show similar sensor behaviour near failure**.

A good degradation sensor should satisfy two conditions:

1. The sensor value should **change significantly during degradation**
2. The **failure values should be similar across different engines**

---

## Step 1: Degradation Magnitude

For each engine:

- Record the **sensor value at the beginning**
- Record the **sensor value at the failure cycle**

Compute the degradation magnitude:

\[
|Start - End|
\]

Example using **Sensor S7**:

| Engine | Start | End |
|---|---|---|
|Engine 1|40|80|
|Engine 2|42|78|
|Engine 3|39|82|

Compute degradation magnitude:

Engine 1:

\[
|40 - 80| = 40
\]

Engine 2:

\[
|42 - 78| = 36
\]

Engine 3:

\[
|39 - 82| = 43
\]

Average degradation magnitude:

\[
Mean = \frac{40 + 36 + 43}{3}
\]

\[
Mean = 39.67
\]

This shows that the sensor **changes significantly during degradation**.

---

## Step 2: Failure Value Variation

Failure values:

\[
[80,78,82]
\]

Mean failure value:

\[
Mean = \frac{80+78+82}{3} = 80
\]

Standard deviation is then calculated to measure variation between engines.

Small standard deviation means **engines fail at similar sensor values**, which is desirable.

---

## Step 3: Prognosability Formula

The prognosability score is computed as:

\[
Prognosability = e^{-\frac{\sigma_{failure}}{\mu_{degradation}}}
\]

Where:

- \(\sigma_{failure}\) = standard deviation of failure values  
- \(\mu_{degradation}\) = mean degradation magnitude  

Example result:

\[
Prognosability \approx 0.96
\]

### Interpretation

- Sensor changes significantly during degradation  
- Failure values are consistent across engines  

Therefore the sensor is **reliable for RUL prediction**.

---

# 3️⃣ Final Sensor Scoring

The final sensor score combines **monotonicity** and **prognosability**.

\[
SensorScore = 0.5 \times Monotonicity + 0.5 \times Prognosability
\]

Both metrics are given equal importance.

---

## Example with Turbofan Sensors

| Sensor | Monotonicity | Prognosability |
|---|---|---|
|S7|0.92|0.88|
|S11|0.75|0.70|
|S3|0.30|0.40|

### Sensor S11 Score

\[
Score = 0.5(0.75) + 0.5(0.70)
\]

\[
Score = 0.375 + 0.35 = 0.725
\]

### Sensor S3 Score

\[
Score = 0.5(0.30) + 0.5(0.40)
\]

\[
Score = 0.15 + 0.20 = 0.35
\]

---

## Sensor Ranking

| Sensor | Score | Rank |
|---|---|---|
|S7|0.90|1|
|S11|0.725|2|
|S3|0.35|3|

### Interpretation

- **S7 → Excellent degradation sensor**
- **S11 → Moderately useful**
- **S3 → Poor sensor**

---

## Final Sensor Selection

The system selects **top-ranked sensors** based on the final score.

These selected sensors are then used for:

- Sliding window sequence creation
- Deep learning model training
- Remaining Useful Life prediction

## Sliding Window Technique

The dataset is converted into **fixed-length sequences** using a sliding window approach.

Example:

Cycle 1–30 → Input Sequence 1  
Cycle 2–31 → Input Sequence 2  
Cycle 3–32 → Input Sequence 3  

This allows the model to learn **temporal degradation patterns** over multiple cycles.

---

# 🧠 Model Architecture

The proposed deep learning architecture combines:

- Convolutional Neural Networks (CNN)
- Bidirectional Long Short-Term Memory (BiLSTM)
- Attention Mechanism
- Fully Connected Layer

Overall workflow:

Sensor Data → Sliding Window → CNN → BiLSTM → Attention → Dense Layer → RUL Prediction

This architecture captures both **local patterns** and **long-term temporal dependencies**.

---

# 1️⃣ Convolutional Neural Network (CNN)

<p align="center">
<img src="images/cnn.png.jpeg" width="600">
</p>

A **1D Convolutional Neural Network** is used to extract local features from sensor sequences.

### Convolution Operation

The convolution operation can be represented as:

$$
y(t) = \sum_{i=0}^{k} x(t-i) \cdot w(i)
$$

Where:

- $x(t)$ = input signal at time step $t$  
- $w(i)$ = convolution filter weights  
- $k$ = filter size  

The convolution layer produces **feature maps** that highlight important degradation patterns.

CNN helps capture **short-term changes in sensor readings**, which are early indicators of engine degradation.

---

# 2️⃣ Bidirectional Long Short-Term Memory (BiLSTM)

The features extracted by CNN are passed to a **Bidirectional Long Short-Term Memory (BiLSTM)** network.

LSTM networks use memory cells and gates to capture long-term dependencies.

### Forget Gate

Determines which previous information should be discarded.

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

### Input Gate

Determines which new information should be stored.

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$

### Candidate Memory

$$
\tilde{C_t} = tanh(W_c[h_{t-1}, x_t] + b_c)
$$

### Memory Update

$$
C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C_t}
$$

### Output Gate

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$

### Hidden State

$$
h_t = o_t \cdot tanh(C_t)
$$

---

### Bidirectional Processing

In BiLSTM, the sequence is processed in both directions:

- Forward: $\overrightarrow{h_t}$
- Backward: $\overleftarrow{h_t}$

Final output:

$$
h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]
$$

This allows the model to capture **complete temporal context**.

---

# 3️⃣ Attention Mechanism

The attention layer helps the model focus on the **most important time steps**.

### Alignment Score

$$
e_t = v^T tanh(W_h h_t + b)
$$

### Attention Weights

$$
\alpha_t = \frac{exp(e_t)}{\sum_{i=1}^{T} exp(e_i)}
$$

### Context Vector

$$
c = \sum_{t=1}^{T} \alpha_t h_t
$$

The context vector represents a **weighted summary of the sequence**, focusing on important degradation patterns.

---

# 4️⃣ Fully Connected Layer (RUL Prediction)

The context vector from the attention layer is passed to a dense layer to predict RUL.

$$
RUL = Wc + b
$$

Where:

- $c$ = context vector  
- $W$ = weight matrix  
- $b$ = bias  

The output is the **predicted number of cycles remaining before engine failure**.

---

# 📊 Results and Discussion

<p align="center">
<img src="images/image1.png.jpeg" width="600">
</p>
<p align="center">
<img src="images/image2.png.jpeg" width="600">
</p>
<p align="center">
<img src="images/image3.png.jpeg" width="600">
</p>

PYTHON RESULTS:
<p align="center">
<img src="images/image4.png.jpeg" width="600">
</p>
<p align="center">
<img src="images/image5.png.jpeg" width="600">
</p>
<p align="center">
<img src="images/image6.png.jpeg" width="600">
</p>
<p align="center">
<img src="images/image7.png.jpeg" width="600">
</p>
<p align="center">
<img src="images/image8.png.jpeg" width="600">
</p>
<p align="center">
<img src="images/image9.png.jpeg" width="600">
</p>

Matlab Results: 
<p align="center">
<img src="images/image10.png.jpeg" width="600">
</p>
<p align="center">
<img src="images/image11.png.jpeg" width="600">
</p>
<p align="center">
<img src="images/image12.png.jpeg" width="600">
</p>
<p align="center">
<img src="images/image13.png.jpeg" width="600">
</p>
The proposed deep learning model shows strong performance in predicting Remaining Useful Life.

Key observations:

- Predicted RUL values closely follow **true degradation trends**
- CNN captures **local sensor patterns**
- BiLSTM models **long-term dependencies**
- Attention improves **prediction accuracy and interpretability**

Evaluation metrics used:

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score

*(Detailed results and graphs will be added below.)*

---

# 🔮 Future Plans

Future improvements include:

- Deploying the system for **real aircraft maintenance environments**
- Extending the approach to other systems such as:
  - Wind turbines
  - Industrial machinery
  - Electric vehicle batteries
- Studying robustness under **noisy sensor conditions**
- Developing a **lightweight real-time monitoring system**

---

# 📚 References

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).  
Damage propagation modeling for aircraft engine run-to-failure simulation.  
NASA Ames Research Center.

https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Dida, M., Cheriet, A., & Belhadj, M. (2025).  
Remaining Useful Life Prediction Using Attention-LSTM Neural Network of Aircraft Engines.  
International Journal of Prognostics and Health Management.

https://www.phmpapers.org

Ferreira, L., & Gonçalves, R. (2022).  
Remaining Useful Life Estimation Using Deep Learning and the NASA C-MAPSS Dataset.  
Scientific Reports, Springer Nature.

https://www.nature.com
