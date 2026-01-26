# 🚀 D7_MFC4_RUL Prediction And PCA  
## ✈️ Attention-Based Remaining Useful Life (RUL) Prediction for Aircraft Turbofan Engines using PCA

## 📌 Project Title

**Attention-Based Remaining Useful Life Prediction for Aircraft Turbofan Engines using PCA and Deep Learning**

## 👥 Team Members

- **Poornima P** – cb.sc.u4aie24343 – poornimap@example.com  
- **Ch. Sarvani Sruthi** – cb.sc.u4aie24311 – sarvani@example.com  
- **Shri Manasa** – cb.sc.u4aie24356 – manasa@example.com  
- **Sowmya A** – cb.sc.u4aie24357 – sowmya@example.com  

## 🎯 Objective

The objective of this project is to predict the Remaining Useful Life (RUL) of aircraft turbofan engines using multivariate time-series sensor data. The aim is to enable predictive maintenance by estimating how long an engine can operate before failure, thereby improving safety, reducing downtime, and minimizing maintenance costs.

## 💡 Motivation / Why the Project is Interesting

Aircraft engine maintenance is highly critical and expensive. Traditional maintenance strategies either replace components too early or too late, leading to increased costs or safety risks. This project is interesting because it uses real-world NASA engine degradation data and combines statistical dimensionality reduction with deep learning. PCA simplifies complex sensor data, while the attention mechanism allows the model to focus on the most critical degradation periods.

## 🛠️ Methodology

### 📂 Dataset

The NASA C-MAPSS turbofan engine dataset is used in this project. It contains multivariate time-series sensor data collected from multiple engines operating under different conditions until failure. Experiments are conducted on FD002, FD003, and FD004 sub-datasets.

### 🔄 Data Preprocessing

- Normalization of sensor values  
- Sliding window technique to convert time-series data into fixed-length sequences suitable for deep learning models  

### 📉 Feature Reduction using PCA (Only)

Principal Component Analysis (PCA) is used as the **only feature reduction technique** in this project.

- Reduces high-dimensional sensor data  
- Removes redundancy and noise  
- Retains maximum variance related to engine health  

📌 No other sensor selection or filtering methods are used.

### 📐 Mathematical Technique Used – PCA

PCA involves centering the data, computing the covariance matrix, extracting eigenvalues and eigenvectors, and projecting the original data onto the top principal components that capture maximum variance.

### 🧸 Toy Example (PCA Demonstration)

Consider two highly correlated sensors:

- Sensor 1: [2, 3, 4]  
- Sensor 2: [4, 6, 8]

PCA combines these two sensors into a single principal component that captures most of the information, reducing dimensionality while preserving the degradation trend.

### 🧠 Model Architecture

The reduced PCA features are fed into a deep learning model consisting of **CNN + BiLSTM + Attention** layers.

- **CNN (1D Convolution):** Extracts short-term degradation patterns  
- **BiLSTM:** Captures long-term temporal dependencies in both forward and backward directions  
- **Attention Layer:** Assigns higher importance to critical degradation time steps  
- **Fully Connected Layer:** Predicts the Remaining Useful Life (RUL)

## 📊 Results and Discussion

The model shows strong RUL prediction performance across all evaluated datasets. Predicted RUL values closely follow the true RUL trends. PCA effectively reduces sensor dimensionality while preserving important degradation information, and the attention mechanism improves prediction accuracy and interpretability.

Evaluation metrics used:
- RMSE  
- MAE  
- R² Score  

## 🔮 Future Plans

- Deploy the PCA-based RUL prediction system in real aircraft maintenance scenarios  
- Extend the approach to wind turbines, industrial machinery, and electric vehicle batteries  
- Study robustness under noisy sensor conditions  
- Develop a lightweight real-time monitoring system  

## 📚 References

1. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).  
   *Damage propagation modeling for aircraft engine run-to-failure simulation*, NASA Ames Research Center.  
   https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

2. Dida, M., Cheriet, A., & Belhadj, M. (2025).  
   *Remaining Useful Life Prediction Using Attention-LSTM Neural Network of Aircraft Engines*.  
   International Journal of Prognostics and Health Management.  
   https://www.phmpapers.org

3. Ferreira, L., & Gonçalves, R. (2022).  
   *Remaining Useful Life Estimation Using Deep Learning and the NASA C-MAPSS Dataset*.  
   Scientific Reports, Springer Nature.  
   https://www.nature.com
