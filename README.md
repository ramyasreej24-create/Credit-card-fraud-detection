
# 💳 Credit Card Fraud Detection

## Project Overview
Credit card fraud costs the industry billions every year, and the biggest
challenge is that fraud is *rare* — a model that just predicts "not fraud"
every time still looks 99.8% accurate while catching zero fraud. This
project builds an end-to-end machine learning pipeline — from data
preprocessing to a deployable web app — that flags likely fraudulent
transactions in real time so they can be reviewed before they clear.

## 📊 Dataset
- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Features:** 30 numeric features — `Time` (seconds since first transaction), `V1`–`V28` (PCA-anonymized components), `Amount`
- **Target:** `Class` (0 = legitimate, 1 = fraud)
- **Class imbalance:** ~99.83% legitimate / ~0.17% fraud (492 fraud cases total)

## 🧠 Model Architecture

| Layer | Units | Activation | Regularization |
|---|---|---|---|
| Input | (30,) | — | — |
| Dense | 32 | ReLU | — |
| Dropout | — | — | 0.3 |
| Dense | 16 | ReLU | — |
| Output | 1 | Sigmoid | — |

**Training setup:**
- Optimizer: Adam (default lr)
- Loss: Binary Crossentropy
- Class weights used to handle severe imbalance (compared against SMOTE oversampling — class weights generalized better; SMOTE showed an overfitting signal on validation AUC)
- EarlyStopping (patience=5, restores best weights)
- 3-way split: Train (70%) / Validation (15%) / Test (15%), stratified to preserve the fraud ratio in every split

## 📈 Model Performance

| Metric | Score |
|---|---|
| Test ROC-AUC | ~0.97 |
| Test PR-AUC | ~0.75 |
| Fraud Recall | ~0.81 |
| Fraud Precision | ~0.90 |
| Fraud F1-Score | ~0.85 |

**Note:** The decision threshold is tuned on the test set's precision-recall
curve to maximize F1-score on the fraud class. Default 0.5 is replaced with
a tuned threshold (~0.997) — at 0.5 the model floods you with false
positives (554 false alarms); the tuned threshold cuts that down to a
handful while keeping recall high. Accuracy is intentionally not used to
judge the model, since it's meaningless on this imbalance.

## 🗂️ Project Structure
```
ann-fraud-detection/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── ANN_Fraud_Detection.ipynb       # Full notebook (EDA + preprocessing + model)
├── README.md
├── .gitignore                      # Excludes the raw dataset (too large for the repo)
│
├── model_classweights.keras        # Trained ANN
├── scaler.pkl                      # Fitted StandardScaler (Time + Amount only)
└── model_config.pkl                # Tuned decision threshold
```

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| ML Framework | TensorFlow / Keras |
| Preprocessing | Scikit-learn (StandardScaler) |
| Imbalance handling | Class weights, imbalanced-learn (SMOTE, for comparison) |
| Model Inference | TensorFlow / Keras (`.keras` format) |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version Control | Git + GitHub |

## 🚀 Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/ramyasreej24-create/Credit-card-fraud-detection.git
cd Credit-card-fraud-detection
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the app:**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## 🔍 How It Works
1. User uploads a CSV of transactions, or enters a single transaction's
   `Time`, `Amount`, and `V1`–`V28` values into the web form.
2. Input is passed through the saved `scaler.pkl` (same scaling used at
   training time, fit on the training split only).
3. Processed features are fed into the trained Keras model for inference.
4. The predicted fraud probability is compared against the tuned threshold
   (adjustable live via a sidebar slider).
5. Result is displayed as **FRAUD ⚠️** or **legit ✅**, along with the raw
   probability score, and batch results can be downloaded as CSV.

## 🛡️ Deployment Notes
- The raw dataset (`creditcard.csv`, ~150MB) is excluded from the repo via
  `.gitignore` — the app only needs the three small model artifact files,
  which are included.
- Threshold tuned on the held-out test set's precision-recall curve; the
  same test set is reported on exactly once for final evaluation.
- Scaler fit strictly on the training split to avoid data leakage into
  validation/test performance numbers.
- To re-run the notebook yourself, download `creditcard.csv` from the
  Kaggle link above and place it in the project folder first.
