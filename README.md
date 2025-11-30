# machinelearning-project
#Team members
Nithin Gundapu :700772575





# 🧠 Breast Cancer Detection: ML vs DL vs QML

## 📌 Project Overview
This project presents a comparative analysis of classical Machine Learning (ML), Deep Learning (DL), and Quantum Machine Learning (QML) algorithms for early-stage breast cancer detection. Using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, we evaluate multiple models across accuracy, precision, recall, and F1-score to identify the most effective approach for binary classification (malignant vs benign).

---

## 📊 Dataset
- Source: [UCI ML Repository – WDBC](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Samples: 569
- Features: 30 numeric attributes (e.g., radius_mean, texture_mean, perimeter_mean)
- Labels: Diagnosis (M = malignant, B = benign)
---

## ⚙️ Methodology

### 🔧 Preprocessing
- Label encoding (M → 1, B → 0)
- Feature scaling using StandardScaler
- Train/test split (75/25)

### 🧪 Algorithms Compared

#### Classical ML
- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)
- Support Vector Machine (SVM – Linear & RBF)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes (GNB)

#### Deep Learning
- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)

#### Quantum ML
- Quantum Support Vector Classifier (QSVC)
- Variational Quantum Classifier (VQC)

---

## 📈 Visualizations
1. Bar chart: Accuracy comparison of ML models  
2. Heatmap: Feature correlation matrix  
3. Box plot: Radius mean distribution by diagnosis  
4. Pairplot: Feature separation by diagnosis  
5. Confusion matrices for LR, CNN, QSVC

---

## 📖 Key Insights
- CNN achieved highest accuracy: **98.6%**
- LR and SVM (Linear) performed best among classical ML: **97.9%**
- QSVC reached **93%**, limited by quantum noise and hardware constraints
- Radius, perimeter, and area features showed strong correlation
- Malignant tumors had higher radius_mean values

---

## 📦 Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn qiskit tensorflow keras
