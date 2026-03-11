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

Each dataset contains the following information:

- Engine ID  
- Cycle number  
- Operational settings  
- Multiple sensor measurements  

Each row represents **one operating cycle of a specific engine**.

As the number of cycles increases, the engine gradually degrades until failure.

---

# 🔄 Data Preprocessing

Before training the deep learning model, several preprocessing steps are applied to prepare the sensor data.

## Normalization

Sensor values may have different numerical ranges. To ensure stable model training, all sensor values are **normalized to a common scale**.

Normalization helps to:

- Improve training stability  
- Prevent large-value sensors from dominating the learning process  
- Improve convergence of the deep learning model  

---

## Sliding Window Technique

The dataset is a **time-series dataset**, meaning the sensor readings change over time. To capture temporal patterns, a **sliding window technique** is used.

This technique converts the time-series data into **fixed-length sequences**.

Example:

Cycle 1–30 → Input Sequence 1  
Cycle 2–31 → Input Sequence 2  
Cycle 3–32 → Input Sequence 3  

Each sequence represents the **recent behaviour of the engine over several cycles**, allowing the model to learn how degradation evolves over time.

---

# 🧠 Model Architecture

To accurately estimate the Remaining Useful Life of the engines, a **hybrid deep learning architecture** is used.

The model combines:

- Convolutional Neural Networks (CNN)
- Bidirectional Long Short-Term Memory (BiLSTM)
- Attention Mechanism

Overall workflow:

Sensor Data → Sliding Window Sequences → CNN → BiLSTM → Attention → Dense Layer → RUL Prediction

This architecture captures both **local degradation patterns** and **long-term temporal dependencies**.

---

# 1️⃣ Convolutional Neural Network (CNN)

The first component of the model is a **1D Convolutional Neural Network (CNN)**.

CNN is used to automatically extract meaningful features from the multivariate sensor sequences.

### Role of CNN

The CNN layer helps to:

- Capture **local temporal patterns in sensor signals**
- Detect **short-term degradation trends**
- Extract important features automatically from raw sensor data

Instead of manually designing features, CNN learns relevant patterns directly from the data.

### Convolution Operation

In a convolution operation, a filter slides across the input sequence and performs element-wise multiplication.

This operation produces a **feature map** that highlights important patterns in the data.

CNN is effective because engine degradation often appears as **small local variations in sensor readings**, which convolution filters can detect.

---

# 2️⃣ Bidirectional Long Short-Term Memory (BiLSTM)

After CNN extracts local features, the feature maps are passed to a **Bidirectional Long Short-Term Memory (BiLSTM)** network.

BiLSTM is a type of **Recurrent Neural Network (RNN)** designed to learn **long-term dependencies in sequential data**.

### Why LSTM is Needed

Engine degradation occurs gradually across many cycles. Therefore, the model must understand **long-term temporal relationships** between sensor readings.

LSTM networks solve this problem using **memory cells and gating mechanisms** that regulate the flow of information through time.

### Bidirectional Learning

In **Bidirectional LSTM**, the sequence is processed in two directions:

- Forward direction (past → future)
- Backward direction (future → past)

This allows the model to learn **complete temporal context of engine degradation**, improving prediction performance.

---

# 3️⃣ Attention Mechanism

After the BiLSTM layer processes the sequence, an **Attention Layer** is applied.

The attention mechanism allows the model to **focus on the most important time steps in the sensor sequence**.

Not all cycles contribute equally to predicting failure. Some cycles contain stronger degradation indicators.

### How Attention Helps

The attention layer assigns **different importance weights to different time steps**.

This allows the model to focus on critical degradation periods that are most useful for predicting RUL.

Advantages of Attention:

- Improves prediction accuracy  
- Highlights important degradation cycles  
- Makes the model more interpretable  

---

# 4️⃣ Fully Connected Layer (RUL Prediction)

The final stage of the network is a **Fully Connected (Dense) Layer**.

The information extracted by CNN, BiLSTM, and the Attention mechanism is passed to this layer, which produces the final **Remaining Useful Life (RUL)** prediction.

The output is a **single numerical value representing the number of cycles remaining before engine failure**.

---

# 📊 Results and Discussion

The proposed deep learning model shows **strong performance in predicting Remaining Useful Life across the evaluated datasets**.

Key observations include:

- Predicted RUL values closely follow the **true degradation trends**
- CNN captures **local sensor patterns**
- BiLSTM models **long-term temporal dependencies**
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
- Extending the approach to other industrial systems such as:
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
