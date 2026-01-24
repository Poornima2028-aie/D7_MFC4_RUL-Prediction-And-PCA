# D7_MFC4_RobustPCA
This project focuses on predictive maintenance of aircraft turbofan engines by estimating their Remaining Useful Life (RUL) using data-driven modeling techniques. The objective is to move beyond traditional reactive and schedule-based maintenance strategies by effectively utilizing sensor data collected during engine operation.
The study uses the NASA C-MAPSS turbofan engine dataset, which consists of multivariate time-series sensor measurements recorded under different operating and fault conditions. The proposed approach aims to identify informative degradation patterns and accurately predict RUL by modeling temporal behavior in the sensor data.

Key Features
Data-driven sensor selection based on:
Variance-based filtering
Monotonicity measure
Prognosability measure
Sliding window–based time-series formulation
Hybrid CNN–BiLSTM–Attention architecture for RUL prediction:
CNN layers for local temporal feature extraction
Bidirectional LSTM for capturing long-term temporal dependencies
Attention mechanism to emphasize critical degradation periods
Performance evaluation using RMSE, MAE, R² score, and tolerance-based accuracy
Implementation and result analysis carried out in both Python and MATLAB

Model Architecture

The model architecture consists of one-dimensional convolutional layers followed by max pooling to extract short-term degradation features from sensor sequences. A Bidirectional LSTM layer is then used to model long-term temporal dependencies by processing the sequence in both forward and backward directions. An attention layer assigns different importance weights to time steps, enabling the model to focus on the most relevant degradation information. Fully connected layers are used to perform final regression and predict the Remaining Useful Life.

Dataset

The NASA C-MAPSS dataset is used in this project. The experiments are conducted on the FD002, FD003, and FD004 sub-datasets, which include varying operating conditions, fault modes, and environmental variability. Each dataset contains multiple sensor measurements recorded across the full operational life of turbofan engines until failure.

Results

The experimental results demonstrate strong RUL prediction performance across all evaluated datasets. Predicted RUL values closely follow the true RUL trends, indicating good agreement between model outputs and actual engine degradation behavior. Error distributions show minimal bias, and the evaluation metrics confirm the reliability and accuracy of the proposed approach.

Future Scope

Future work will focus exclusively on the use of Principal Component Analysis (PCA) for feature reduction and degradation modeling. The aim is to analyze the effectiveness of PCA in capturing engine health information while reducing data dimensionality and sensor redundancy. No additional deep learning architectures will be explored, allowing for a focused study on PCA-based RUL prediction and its robustness under varying operating conditions.
