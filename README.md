# Autoencoder for Static Anomaly Detection

This repository implements an **unsupervised autoencoder-based anomaly detection system** for identifying fraudulent credit card transactions using **TensorFlow/Keras**.  
The model is trained exclusively on **non-fraud (normal) transactions** and detects anomalies using **reconstruction error**.



## 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions are rare.  
In such scenarios, **unsupervised learning** techniques like autoencoders are effective because they learn normal transaction patterns without relying on labeled fraud data.

This project:
- Trains an autoencoder only on normal transactions
- Uses reconstruction error as an anomaly score
- Applies a percentile-based threshold to classify fraud
- Evaluates performance using metrics suitable for imbalanced data



## 🛠️ Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- Pandas  
- scikit-learn  
- Matplotlib  



## 📂 Dataset

- **Credit Card Fraud Detection Dataset** from Kaggle
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Contains anonymized transaction features  
- Highly imbalanced (fraud transactions are rare)



## ⚙️ Methodology

### 1. Data Preprocessing
- Removed target label from input features
- Normalized features using `StandardScaler`
- Split data to train autoencoder only on non-fraud samples

### 2. Autoencoder Model
- Fully connected (Dense) encoder–decoder architecture
- Bottleneck layer to learn compressed representation
- Mean Squared Error (MSE) loss for reconstruction

### 3. Anomaly Detection
- Reconstruction error used as anomaly score
- Threshold selected as **95th percentile** of reconstruction error on normal transactions

### 4. Evaluation
- Confusion Matrix
- Precision, Recall, F1-score
- ROC AUC score
- ROC Curve
- Precision–Recall Curve



## 📊 Results

- **High fraud recall**, prioritizing detection of fraudulent transactions  
- **ROC AUC ≈ 0.93–0.95**, indicating strong separation between normal and fraud samples  
- Precision–Recall analysis highlights expected trade-offs due to class imbalance



## 📈 Visualizations Included

- Training & validation loss curves  
- Reconstruction error distribution (log scale)  
- Confusion matrix  
- ROC curve  
- Precision–Recall curve  



## 🧠 Key Takeaways

- Autoencoders are effective for anomaly detection in imbalanced datasets
- Percentile-based thresholding provides flexible control over false positives and recall
- Recall is prioritized over precision in fraud detection scenarios



## 🚀 Future Improvements

- Threshold optimization based on business cost
- Variational Autoencoders (VAE)
- Comparison with Isolation Forest or One-Class SVM
- Deployment as a real-time fraud detection service



## 📌 Author

**Dasari Manasa Sowmya**



