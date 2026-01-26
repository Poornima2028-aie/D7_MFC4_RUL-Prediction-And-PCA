Attention-Based Remaining Useful Life (RUL) Prediction for Aircraft Turbofan Engines using PCA and Deep Learning

👥 Team Members
Name	Roll Number	Email
Poornima P	cb.sc.u4aie24343	poornimap@example.com

Ch. Sarvani Sruthi	cb.sc.u4aie24311	sarvani@example.com

Shri Manasa	cb.sc.u4aie24356	manasa@example.com

Sowmya A	cb.sc.u4aie24357	sowmya@example.com
🎯 Objective

The objective of this project is to predict the Remaining Useful Life (RUL) of aircraft turbofan engines using time-series sensor data. The project aims to:

Reduce unexpected engine failures

Enable predictive maintenance

Improve safety and reduce maintenance cost

Study the effectiveness of PCA as the sole feature-reduction technique combined with a deep learning model

💡 Motivation / Why This Project is Interesting

Aircraft engine failures are extremely costly and dangerous. Traditional maintenance approaches either replace parts too early or too late.

This project is interesting because:

It uses real NASA engine data

Combines statistical dimensionality reduction (PCA) with deep learning

Uses an attention mechanism to focus on critical degradation periods

Demonstrates how complex sensor data can be simplified using PCA without losing essential health information

🛠 Methodology
1️⃣ Dataset Used

NASA C-MAPSS Turbofan Engine Dataset

Sub-datasets: FD002, FD003, FD004

Multivariate time-series sensor data until engine failure

2️⃣ Data Preprocessing

Removal of non-informative sensors

Normalization of sensor values

Sliding window approach to convert time-series data into sequences

3️⃣ Feature Reduction using PCA (Only)

Principal Component Analysis (PCA) is used as the only feature reduction technique in this project.

Reduces high-dimensional sensor data

Removes redundancy and noise

Retains maximum variance (engine health information)

📌 No other sensor selection or filtering methods are used.

🔢 Mathematical Technique: PCA (Simple Explanation)

Given a data matrix 
𝑋
X:

Center the data

Compute covariance matrix

Calculate eigenvalues and eigenvectors

Project data onto top-k principal components

🧸 Toy Example (PCA)

Suppose we have 2 sensors:

Sensor 1	Sensor 2
2	4
3	6
4	8

These sensors are highly correlated.

👉 PCA combines them into one principal component that captures most of the information, reducing dimensionality from 2 → 1.

4️⃣ Model Architecture

The reduced PCA features are fed into a deep learning model:

CNN + BiLSTM + Attention

CNN (1D Convolution):
Extracts short-term degradation patterns

BiLSTM:
Captures long-term dependencies in both forward and backward directions

Attention Layer:
Assigns higher importance to critical time steps related to engine failure

Fully Connected Layer:
Predicts the Remaining Useful Life (RUL)

📊 Results & Discussion

Predicted RUL values closely match actual RUL trends

Low prediction error across different operating conditions

PCA successfully reduces dimensionality while preserving degradation information

Attention mechanism improves model interpretability by focusing on critical degradation stages

Evaluation Metrics:

RMSE

MAE

R² Score

🚀 Future Plans

Deploy PCA-based RUL prediction in real aircraft maintenance systems

Apply the approach to:

Wind turbines

Industrial motors

Electric vehicle batteries

Study robustness of PCA under noisy sensor conditions

Develop a lightweight real-time monitoring system

📚 References

NASA C-MAPSS Dataset
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Saxena et al. (2008), NASA Ames
https://ti.arc.nasa.gov/m/project/prognostic-data-repository/

Dida et al. (2025), IJPHM
https://www.phmpapers.org

Ferreira & Gonçalves (2022), Scientific Reports
https://www.nature.com/articles/s41598-022-XXXXX
