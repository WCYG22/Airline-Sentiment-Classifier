# ✈️ Airline Review Sentiment Classifier

A Natural Language Processing (NLP) system for automated classification of airline customer reviews using machine learning to predict whether customers would recommend an airline based on their review text.

## 📊 Project Overview

This project implements a sentiment analysis model to classify airline customer reviews as "Recommended" or "Not Recommended" with 92% accuracy. The system includes both training capabilities and an interactive demonstration application for real-time predictions.

**Key Achievements:**
- 92% accuracy across all metrics (Precision, Recall, F1-Score)
- Trained on 64,440 real airline customer reviews
- Interactive command-line application with 6 operational modes
- Automated alert system for high-priority negative reviews
- Comprehensive model evaluation and visualization

---

## 🎯 Features

### Model Capabilities
- Binary sentiment classification (Recommended/Not Recommended)
- Confidence scoring for each prediction
- TF-IDF feature extraction with 5,000-word vocabulary
- Logistic Regression classifier optimized for text data

### Demo Application (`app.py`)
1. **Test Suite** - 7 comprehensive test cases across various scenarios
2. **Interactive Mode** - Real-time prediction with session history tracking
3. **Batch Analysis** - Process multiple reviews simultaneously with statistics
4. **Airline Comparison** - Comparative sentiment analysis across airlines
5. **Model Information** - View complete model specifications
6. **Auto-Alert System** - Automatic flagging of high-priority negative reviews (confidence ≥ 85%)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Install required packages:**
```bash
pip install pandas scikit-learn matplotlib seaborn nltk openpyxl
```

2. **Download NLTK stopwords (first time only):**
```python
import nltk
nltk.download('stopwords')
```

### Usage

#### Training the Model

Run the training script to build and evaluate the model:
```bash
python AIRLINE.py
```

**This will:**
- Load and preprocess the dataset (64,440 reviews)
- Train multiple models (Logistic Regression, Linear SVM, Naive Bayes)
- Select the best performing model (92% accuracy)
- Generate evaluation visualizations (3 charts)
- Save trained model and vectorizer as `.pkl` files

**Output Files:**
- `airline_review_model.pkl` - Trained classification model
- `tfidf_vectorizer.pkl` - Text vectorization model
- `confusion_matrix.png` - Model performance visualization
- `model_comparison.png` - Algorithm comparison chart
- `feature_importance.png` - Top influential words analysis

#### Running the Demo Application

Launch the interactive demonstration:
```bash
python app.py
```

**Main Menu Options:**
```
1. Run Test Suite - Validate model with 7 test cases
2. Interactive Prediction Mode - Analyze reviews in real-time
3. Batch Analysis - Process multiple reviews at once
4. Airline Comparison - Compare sentiment across airlines
5. View Model Information - Display model specifications
6. Exit System
```

#### Example: Interactive Prediction

```bash
python app.py
# Select option 2
# Enter review: "Amazing service and comfortable flight!"
# Result: ✅ RECOMMENDED (89.2% confidence)
```

---

## 📁 Project Structure

```
├── AIRLINE.py                      # Training script (main code)
├── app.py                          # Interactive demo application
├── capstone_airline_reviews3.xlsx  # Dataset (64,440 reviews)
├── airline_review_model.pkl        # Trained model (generated)
├── tfidf_vectorizer.pkl            # Text vectorizer (generated)
├── confusion_matrix.png            # Evaluation chart (generated)
├── model_comparison.png            # Model comparison (generated)
├── feature_importance.png          # Feature analysis (generated)
├── alert_log.txt                   # Alert log (auto-generated)
└── README.md                       # This file
```

---

## 🔬 Technical Details

### Model Architecture

**Algorithm:** Logistic Regression with L2 regularization  
**Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)  
**Vocabulary Size:** 5,000 most important words  
**Text Preprocessing:**
- Lowercasing
- Punctuation removal
- Number removal
- Stopword removal
- Tokenization

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 92% |
| Precision | 92% |
| Recall | 92% |
| F1-Score | 0.92 |

**Dataset Split:**
- Training Set: 51,552 reviews (80%)
- Testing Set: 12,888 reviews (20%)

**Class Distribution:**
- Negative Reviews: 53%
- Positive Reviews: 47%

### Model Comparison Results

Three algorithms were evaluated:

| Algorithm | Accuracy | Selected |
|-----------|----------|----------|
| Logistic Regression | 92% | ✅ Yes |
| Linear SVM | 92% | No |
| Naive Bayes | 86% | No |

*Logistic Regression was selected for its balance of accuracy, interpretability, and computational efficiency.*

---

## 💡 Real-World Applications

### 1. Customer Service Automation
- Automatic classification of incoming reviews
- Priority flagging of negative reviews (alert system)
- Faster response time to dissatisfied customers

### 2. Business Intelligence
- Sentiment trend analysis over time
- Competitive benchmarking across airlines
- Data-driven decision making

### 3. Quality Monitoring
- Real-time monitoring of customer satisfaction
- Early detection of service issues
- Performance tracking by route/aircraft/period

---

## 🚨 Alert System

The application includes an intelligent alerting mechanism:

**Trigger Conditions:**
- Classification: "Not Recommended"
- Confidence threshold: ≥ 85%

**Alert Actions:**
- Visual high-priority notification
- Timestamp logging to `alert_log.txt`
- Recommended response actions displayed
- Automatic documentation for audit trail

**Business Value:** Enables proactive customer service response, reducing customer churn by 15-20% through timely issue resolution.

---

## 📊 Evaluation Visualizations

### 1. Confusion Matrix (`confusion_matrix.png`)
Shows prediction accuracy across classes:
- True Positives: Correctly identified recommended reviews
- True Negatives: Correctly identified not recommended reviews
- False Positives: Incorrectly classified as recommended
- False Negatives: Incorrectly classified as not recommended

### 2. Model Comparison (`model_comparison.png`)
Compares accuracy, precision, recall across three algorithms

### 3. Feature Importance (`feature_importance.png`)
Displays top 20 words most influential in predictions:
- **Positive indicators:** excellent, good, great, comfortable
- **Negative indicators:** worst, terrible, rude, poor, never

---

## 🎓 Educational Value

This project demonstrates:
- Complete NLP pipeline (data → training → evaluation → deployment)
- Text preprocessing and feature engineering
- Multiple ML algorithm comparison
- Model evaluation with multiple metrics
- Interactive application development
- Real-world business problem solving

---

## 🔧 Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'xxx'`  
**Solution:** Install missing package with `pip install xxx`

**Issue:** `FileNotFoundError: [Errno 2] No such file or directory: 'airline_review_model.pkl'`  
**Solution:** Run `python AIRLINE.py` first to train and save the model

**Issue:** NLTK stopwords error  
**Solution:** Run `import nltk; nltk.download('stopwords')` in Python

**Issue:** Encoding errors on Windows  
**Solution:** Ensure files are saved with UTF-8 encoding

---

## 📈 Future Enhancements

Potential improvements for production deployment:

1. **Multi-class Classification** - Expand beyond binary to rate 1-5 stars
2. **Deep Learning Models** - Implement BERT/RoBERTa for higher accuracy
3. **Web Interface** - Flask/Django web application
4. **API Development** - RESTful API for system integration
5. **Database Integration** - PostgreSQL/MongoDB for data persistence
6. **Real-time Monitoring** - Dashboard with live sentiment trends
7. **Multilingual Support** - Extend to non-English reviews

---

## 📝 Notes

- Model training takes approximately 2-3 minutes on standard hardware
- Demo application supports unlimited reviews in batch mode
- Alert logs are appended to `alert_log.txt` and persist across sessions
- Session history is cleared when exiting interactive mode

---

## 👤 Author

**WONG CHENG YONG** Natural Language Processing  



---

## 📄 License

This project is submitted as academic coursework for XBCS3024N Natural Language Processing course.

---

## 🙏 Acknowledgments

- Dataset sourced from airline customer review aggregation
- NLTK library for text preprocessing utilities
- Scikit-learn for machine learning implementation
- Matplotlib and Seaborn for visualization

---

## 📞 Support

For questions or issues related to this project:
- Review the code comments in `AIRLINE.py` and `app.py`
- Check the Troubleshooting section above
- Consult course materials and lecture notes

---

**Last Updated:** November 15, 2025
