# ✈️ Airline Sentiment Analysis System

## 📌 Project Overview
Advanced NLP-based sentiment analysis system for airline customer reviews. Predicts whether customers recommend an airline based on their text reviews using machine learning.

---

## 🎯 Key Features

### 1. **Machine Learning Pipeline**
- Text preprocessing with stopword removal and punctuation cleaning
- TF-IDF vectorization for feature extraction
- Multiple model comparison (Logistic Regression, Naive Bayes, SVM)
- Best model selection based on F1 score

### 2. **Interactive Web Dashboard** 
- Real-time single review prediction with confidence scores
- Batch CSV file analysis with downloadable results
- Visual analytics and model performance insights
- Professional UI with gradient design

### 3. **Alert System**
- Automatically flags high-confidence negative reviews (≥85%)
- Logs critical feedback to `alert_log.txt` with timestamps
- Helps identify urgent customer service issues

---

## 📁 Project Structure

```
NLP (ASSIGN)/
├── AIRLINE.py                      # Main training script
├── streamlit_app.py                # Web interface application
├── alert_log.txt                   # Auto-generated alert log
├── airline_review_model.pkl        # Trained ML model
├── tfidf_vectorizer.pkl            # Fitted vectorizer
├── capstone_airline_reviews3.xlsx  # Training dataset
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (Optional)
If you need to retrain from scratch:
```bash
python AIRLINE.py
```
This generates:
- `airline_review_model.pkl` 
- `tfidf_vectorizer.pkl`
- Performance visualizations (confusion matrix, model comparison charts)

### Step 3: Run the Web Application
```bash
streamlit run streamlit_app.py
```
The app opens automatically at `https://airlinereview.streamlit.app/`

---

## 💻 Usage Guide

### Dashboard Overview
- View model accuracy, precision, recall, F1 score
- See training dataset statistics
- Monitor system status

### Prediction Engine
- Enter a customer review
- Get instant sentiment prediction
- View confidence percentage
- Critical negative reviews (≥85% confidence) are auto-logged

### Batch Analysis
- Upload CSV file with `review` column
- Process multiple reviews at once
- Download results with predictions and confidence scores

### Model Insights
- View feature importance (top positive/negative indicators)
- Analyze confusion matrix
- Compare model performance metrics

---

## 📊 Model Performance
- **Algorithm**: Logistic Regression (selected via F1 optimization)
- **Accuracy**: ~95%+ (varies by dataset split)
- **Features**: TF-IDF vectorization with stopword removal

---

## 🔔 Alert Log System

**Purpose**: Track high-priority negative feedback  
**Location**: `alert_log.txt`  
**Trigger**: Reviews with "NOT RECOMMENDED" prediction and ≥85% confidence

**Log Format**:
```
======================================================================
ALERT FLAGGED: 2025-11-21 02:00:08
Review: Worst and rude staff
Prediction: NOT RECOMMENDED
Confidence: 100.0%
======================================================================
```

---

## 🛠️ Technical Stack
- **ML/NLP**: scikit-learn, NLTK, pandas
- **Vectorization**: TF-IDF 
- **UI Framework**: Streamlit
- **Visualization**: matplotlib, seaborn
- **Data Format**: Excel (training), CSV (batch input)

---

## 📝 Notes
- The model is pre-trained on airline review data (64,440+ reviews)
- Text preprocessing includes lowercase conversion, punctuation removal, and stopword filtering
- Alert logs append new entries automatically
- Close and reopen `alert_log.txt` to see new entries
- For production use, consider implementing database storage instead of text file logging

---

## 🐛 Troubleshooting

**Issue**: Alert log not updating in VS Code  
**Solution**: Close and reopen `alert_log.txt` file. The log updates after each alert but may require refreshing in the editor

**Issue**: Model not loading  
**Solution**: Retrain using `AIRLINE.py` to regenerate `.pkl` files

**Issue**: Streamlit errors  
**Solution**: Run `pip install --upgrade streamlit` and restart

---

## 📧 Support
For questions or issues, refer to the project documentation or contact the development team.

---

**Last Updated**: November 2025  
**Version**: 1.0
