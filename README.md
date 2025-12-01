# machinelearning-project
#Team members
Nithin Gundapu :700772575






🧬 Breast Cancer Detection Using Machine Learning, Deep Learning & Quantum Machine Learning
A Comparative Study using the Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
📌 Overview

This project presents a complete comparative analysis of:

Machine Learning (ML)

Deep Learning (DL)

Quantum Machine Learning (QML)

for early-stage breast cancer detection, using the WDBC dataset and an extended 50,000-sample synthetic dataset.
The goal is to identify which AI technique delivers the highest accuracy, precision, and reliability for tumor classification.

🎯 Objectives

Apply ML, DL, and QML algorithms for breast cancer classification.

Evaluate using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

Identify the most effective algorithm for diagnostic decision support.

Generate a large, realistic 50k synthetic WDBC dataset for advanced testing.

Examine QML limitations such as quantum noise and qubit constraints.

📊 Dataset Description
1️⃣ Original WDBC Dataset

Samples: 569

Features: 30 numerical cell nucleus characteristics

Classes:

M → Malignant (1)

B → Benign (0)

Distribution:

Benign: 62.7%

Malignant: 37.3%

2️⃣ Synthetic WDBC Dataset (50,000 Samples)

Generated using:

Class-conditional multivariate Gaussian modeling

Preserved covariance structure

Realistic distributions for ML/DL training

Used for deep learning stability and large-scale evaluation.

🧠 Algorithms Implemented
🔹 Machine Learning Models
Model	Accuracy
Logistic Regression	97.9%
Linear SVM	97.9%
RBF SVM	96.5%
KNN	96.5%
Random Forest	94.4%
Gaussian NB	92.3%
Decision Tree	92.3%
🔹 Deep Learning Models
Model	Accuracy
Convolutional Neural Network (CNN)	98.6%
Artificial Neural Network (ANN)	97.9%
Recurrent Neural Network (RNN)	95.8%

📌 CNN performed best overall, due to its ability to learn nonlinear biomarker interactions.

🔹 Quantum Machine Learning Models (Qiskit)
Quantum Model	Accuracy
QSVC (Quantum SVM Classifier)	93%
VQC (Variational Quantum Classifier)	78%

QML shows promise but is limited by:

Quantum noise

Qubit limitations

Shallow circuits

Decoherence

🧪 Methodology
Load Data → Preprocess → Encode Labels → Scale Features → 
Train Test Split → Train Models (ML/DL/QML) → Evaluate → Compare

✔ Preprocessing Steps

Removed missing values

Encoded labels (M=1, B=0)

StandardScaler applied

Train-Test split: 75% / 25%

✔ Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

📈 Key Results Summary
Category	Best Algorithm	Accuracy
Machine Learning	Logistic Regression / SVM	97.9%
Deep Learning	CNN	98.6%
Quantum ML	QSVC	93%

🏆 CNN is the overall top performer.

🧪 CNN Architecture Used
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(30,1)),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

🚀 How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run Machine Learning models
python src/ml_models.py

3️⃣ Run CNN (Deep Learning)
python src/dl_cnn.py

4️⃣ Run Quantum Models

Requires Qiskit:

python src/qml_qsvc.py
python src/qml_vqc.py

📦 Requirements

Python ≥ 3.8

scikit-learn

numpy

pandas

tensorflow / keras

qiskit

matplotlib


