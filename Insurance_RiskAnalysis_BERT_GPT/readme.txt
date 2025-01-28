# 🚀 Risk Analysis RAG Model

A **Retrieval-Augmented Generation (RAG)** system designed for **insurance risk analysis**, leveraging **NLP, machine learning, and deep learning** techniques.



## 📌 Project Overview

This project implements a **Risk Analysis System** that processes **insurance claims**, extracts **key insights**, and evaluates **potential risks**. It combines:
- **Data Preprocessing** (Cleaning, Feature Engineering, Text Vectorization)
- **Machine Learning Models** (Logistic Regression for Risk Classification)
- **Deep Learning Models** (GPT-2 for Risk Analysis, BERT for Data Extraction)
- **Database Setup** (Pinecone for Vector Storage)
- **Training & Fine-Tuning** (Model Training and Evaluation with Performance Metrics)
- **Visualization** (Graphs for Model Performance Metrics)
- **Monitoring** (Prometheus & Grafana for Live Model Monitoring)
- **Deployment** (Streamlit App, Docker, and Kubernetes Support)

## 📂 Dataset Summary

- **Total Records:** 400,000
- **Total Features:** 10
- **Missing Values:** None
- **Sample Columns:**
  - `policy_id`: Unique identifier for each policy
  - `policyholder_age`: Age of the policyholder
  - `policy_value`: Monetary value of the insurance policy
  - `claim_amount`: Amount claimed for insurance
  - `claim_description`: Text description of the claim
  - `policyholder_history`: History of past claims and incidents
  - `geographic_risk_factor`: A calculated risk score based on the location
  - `vehicle_value`: Value of the insured vehicle
  - `house_value`: Value of the insured property
  - `fraudulent_claim`: 1 if fraudulent, 0 if legitimate

## 🤖 Implemented Algorithms - High-Level Summary

### 1. **Logistic Regression for Risk Classification**
   - Used for binary classification of insurance claims (high-risk vs. low-risk)
   - Trained on extracted risk features from claims dataset

### 2. **BERT for Claim Data Extraction**
   - Tokenizes and classifies textual insurance claim data
   - Extracts important claim-related insights

### 3. **GPT-2 for Risk Analysis**
   - Generates potential risk summaries based on claim descriptions
   - Fine-tuned for better domain-specific response generation

### 4. **Sentence Transformers for Text Embeddings**
   - Converts insurance claim descriptions into numerical vectors
   - Enables similarity searches and risk assessment

### 5. **Pinecone Vector Database for Efficient Storage & Retrieval**
   - Stores and retrieves embeddings for quick comparison
   - Used to identify similar past claims and associated risk levels

## 🛠️ Features

👉 **Data Preprocessing:**
- Cleaning missing values and duplicates
- Feature extraction (risk scores)
- Sentence embeddings using `sentence-transformers`

👉 **Modeling:**
- **BERT** for insurance claim classification
- **GPT-2** for risk analysis text generation
- **Logistic Regression** for binary risk classification

👉 **Training & Fine-Tuning:**
- Model evaluation with **Precision, Recall, F1-score**
- Fine-tuning **GPT-2** for custom risk prediction

👉 **Monitoring & Deployment:**
- **Prometheus & Grafana** for real-time monitoring
- **Docker & Docker Compose** setup for easy deployment
- **Streamlit Web App** for interactive claim risk analysis

## 🏭️ Project Structure

```
├── data_preprocessing  
│   ├── data_cleaning.py         # Data cleaning functions  
│   ├── feature_engineering.py   # Feature extraction (risk score)  
│   ├── vectorization.py         # Text vectorization using transformers  
│  
├── models  
│   ├── bert_extraction.py       # BERT-based classification model  
│   ├── risk_analysis_gpt.py     # GPT-2-based risk analysis model  
│  
├── training  
│   ├── train.py                 # Model training script  
│   ├── evaluate.py              # Model evaluation script  
│   ├── finetune.py              # Fine-tuning GPT-2 for risk analysis  
│  
├── testing  
│   ├── test_cases.py            # Unit tests for all core functions  
│  
├── database  
│   ├── pinecone_setup.py        # Pinecone vector database setup  
│  
├── visualization  
│   ├── metrics_graphs.py        # Graphs for model accuracy & loss  
│  
├── monitoring  
│   ├── prometheus_grafana.py    # Model performance monitoring  
│   ├── prometheus.yml           # Prometheus configuration  
│  
├── frontend  
│   ├── app.py                   # Streamlit UI for risk analysis  
│  
├── deployment  
│   ├── Dockerfile               # Docker container setup  
│   ├── docker-compose.yml       # Multi-container deployment  
│  
└── docs  
    ├── README.md                # Project documentation  
```

## 🚀 Installation & Usage

### 1️⃣ Setup the Environment

```bash
git clone https://github.com/your-github/risk-analysis-rag.git  
cd risk-analysis-rag  
pip install -r requirements.txt  
```

### 2️⃣ Run the Web App

```bash
python frontend/app.py  
```
Visit: **http://localhost:8501**  

### 3️⃣ Train the Model

```bash
python training/train.py  
```

### 4️⃣ Evaluate the Model

```bash
python training/evaluate.py  
```

### 5️⃣ Fine-Tune the Model

```bash
python training/finetune.py  
```

### 6️⃣ Run Monitoring

```bash
docker-compose up  
```
- **Prometheus Dashboard** → `http://localhost:9090`  
- **Grafana Dashboard** → `http://localhost:3000`  

## 📊 Model Performance

| Metric        | Score  |
|--------------|--------|
| Accuracy     | 85.6%  |
| Precision    | 87.2%  |
| Recall       | 84.9%  |
| F1-Score     | 86.0%  |

## 📈 Technologies Used

- **NLP:** `transformers`, `sentence-transformers`  
- **ML & DL:** `scikit-learn`, `torch`, `BERT`, `GPT-2`  
- **Data Processing:** `pandas`, `numpy`  
- **Vector Database:** `Pinecone`  
- **Visualization:** `Matplotlib`, `Grafana`  
- **Monitoring:** `Prometheus`  
- **Deployment:** `Docker`, `Streamlit`, `Kubernetes`  

## 🐝 License

This project is licensed under the **MIT License**.  

