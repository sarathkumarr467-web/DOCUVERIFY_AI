# 🔐 DocuVerify AI

### Verify Smarter. Detect Tampering. Trust Documents.

DocuVerify AI is an AI-powered document verification and tampering detection system designed to analyze digital documents and identify potentially suspicious or manipulated content.

It supports multiple document formats and combines text analysis, OCR, metadata inspection, and rule-based risk assessment to generate an understandable verification result.

---

## 🚀 Key Features

* 📄 Multi-format document support
* 🔍 Document content analysis
* 🛡️ Tampering and suspicious-content detection
* 📝 OCR-based text extraction
* 🧠 AI/ML-assisted document verification
* 📊 Risk score calculation
* ⚠️ Suspicious keyword and pattern detection
* 📋 Missing-field detection
* 📈 Document verification results
* 💾 Prediction/history storage
* 📑 Verification report generation
* 🎨 User-friendly web dashboard

---

## 📂 Supported File Formats

DocuVerify AI supports:

* PDF
* DOCX
* TXT
* PNG
* JPG
* JPEG

---

## 🏗️ Project Structure

```text
DOCUVERIFY_AI/
│
├── app.py
├── database.py
├── train_model.py
├── requirements.txt
│
├── modules/
│   └── ...
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── dashboard.html
│   ├── prediction.html
│   └── ...
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── model.pkl
├── label_encoder.pkl
├── symptom_list.pkl
│
└── README.md
```

> Update the file names above if your actual repository structure is different.

---

## ⚙️ Technology Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* Flask

### AI / ML

* Machine Learning
* Text Analysis
* OCR
* Pattern Detection
* Metadata Analysis

### Database

* SQLite

### Libraries

* Flask
* Scikit-learn
* Pandas
* NumPy
* Joblib
* ReportLab
* OCR/Text Extraction libraries

---

## 🔄 How It Works

```text
Upload Document
       ↓
File Validation
       ↓
Text / OCR Extraction
       ↓
Content Analysis
       ↓
Metadata Analysis
       ↓
Suspicious Pattern Detection
       ↓
Risk Score Calculation
       ↓
Verification Result
       ↓
Report Generation
```

---

## 📊 Risk Classification

The system calculates a risk score based on different document characteristics.

| Risk Score | Result                      |
| ---------- | --------------------------- |
| 0 – 34     | 🟢 Genuine / Low Risk       |
| 35 – 64    | 🟡 Suspicious / Medium Risk |
| 65 – 100   | 🔴 Fake / High Risk         |

The result is generated based on detected suspicious patterns, keywords, missing information, repeated content, filename indicators, and other document-level signals.

---

## 💡 Why DocuVerify AI?

Digital documents are increasingly used for education, employment, banking, government services, and business transactions.

Manually checking every document can be:

* Time-consuming
* Difficult to scale
* Prone to human error
* Inefficient for large document collections

DocuVerify AI provides an automated first-level verification system that helps users identify potentially suspicious documents quickly.

---

## 🎯 Use Cases

* 🎓 Academic certificate verification
* 💼 Employment document verification
* 🏢 Business document screening
* 🏦 Financial document verification
* 🏛️ Government document screening
* 📑 General digital document validation

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sarathkumarraids2025/DocuVerify-AI.git
```

### 2. Open the Project

```bash
cd DocuVerify-AI
```

If your `app.py` is inside another folder:

```bash
cd DOCUVERIFY_AI
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

The application will start on the local Flask server.

---

## 🌐 Application Workflow

1. Open the DocuVerify AI web application.
2. Upload a supported document.
3. The system extracts and analyzes the document.
4. Suspicious patterns and document characteristics are evaluated.
5. A risk score is generated.
6. The document is classified as Low, Medium, or High Risk.
7. The user can view the verification result.
8. A verification report can be generated.

---

## 🔮 Future Scope

* 🤖 Advanced Deep Learning-based forgery detection
* 🔗 Blockchain-based document verification
* ☁️ Cloud deployment
* 📱 Mobile application
* 🌍 Multi-language document support
* 🔎 Advanced image-level tampering detection
* 🧾 Digital signature verification
* 🔐 Secure document hashing
* 🧠 Explainable AI-based verification insights

---

## 👨‍💻 Developer

**R. Sarathkumar**

B.Tech Artificial Intelligence & Data Science

Chennai Institute of Technology

---

## ⭐ Project Goal

DocuVerify AI aims to make document verification **faster, smarter, explainable, and more accessible** using Artificial Intelligence and Machine Learning.

### 🔐 Verify Smarter. Detect Tampering. Trust Documents.
Live link:
https://docuverify-ai-6.onrender.com/result/23
