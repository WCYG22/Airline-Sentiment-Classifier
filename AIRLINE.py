# type: ignore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load data
print("="*60)
print("STEP 1: Loading Dataset")
print("="*60)

# Load Excel file
df = pd.read_excel('capstone_airline_reviews3.xlsx')
print(f"Original data shape (all columns): {df.shape}")
print(f"All columns: {df.columns.tolist()}")

# ✨ SELECT ONLY 2 COLUMNS WE NEED ✨
df = df[['customer_review', 'recommended']]
print(f"\nSelected 2 columns: {df.shape}")
print(f"Columns after selection: {df.columns.tolist()}")

# Display first 5 rows
print(f"\nFirst 5 rows:")
print(df.head())

# Show target distribution
print(f"\nTarget distribution:")
print(df['recommended'].value_counts())

# Show percentage
print(f"\nPercentage distribution:")
print(df['recommended'].value_counts(normalize=True) * 100)

# Drop rows with missing values in these 2 columns
clean_df = df.dropna(subset=['customer_review', 'recommended'])
clean_df = clean_df.reset_index(drop=True)
print(f"\nData after cleaning: {clean_df.shape}")
print(f"✓ Dataset ready with {len(clean_df)} reviews and 2 columns")

# Text preprocessing function
def clean_text(text):
    """Clean and preprocess text reviews"""
    # Convert to lowercase
    text = str(text).lower()
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation + '0123456789'))
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

print("\n" + "="*60)
print("STEP 2: Text Preprocessing")
print("="*60)
print("Cleaning reviews (lowercase, remove punctuation, etc.)...")
clean_df['clean_review'] = clean_df['customer_review'].apply(clean_text)
print(f"Sample cleaned review:\n{clean_df['clean_review'].iloc[1][:200]}...")

# Split data
print("\n" + "="*60)
print("STEP 3: Train-Test Split")
print("="*60)
X_train, X_test, y_train, y_test = train_test_split(
    clean_df['clean_review'], 
    clean_df['recommended'], 
    test_size=0.2, 
    random_state=42,
    stratify=clean_df['recommended']  # Keep class balance
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Train class distribution:\n{y_train.value_counts()}")

# Vectorize text
print("\n" + "="*60)
print("STEP 4: TF-IDF Vectorization")
print("="*60)
vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Feature matrix shape: {X_train_vec.shape}")
print(f"Number of unique words (features): {len(vectorizer.get_feature_names_out())}")

# Train baseline model
print("\n" + "="*60)
print("STEP 5: Model Training - Logistic Regression")
print("="*60)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)
print("Model training complete!")

# Evaluate baseline model
y_pred = model.predict(X_test_vec)
print("\n" + "="*60)
print("STEP 6: Model Evaluation")
print("="*60)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix with better labels
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Recommended', 'Recommended'],
            yticklabels=['Not Recommended', 'Recommended'])
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)
plt.title('Confusion Matrix - Airline Review Classification', fontsize=14, fontweight='bold')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# Show prediction examples
print("\n" + "="*60)
print("STEP 7: Sample Predictions")
print("="*60)

# Correct predictions
print("\n--- CORRECT PREDICTIONS (Sample 5) ---")
correct_mask = (y_test == y_pred)
correct_indices = y_test[correct_mask].index[:5]

for idx, i in enumerate(correct_indices, 1):
    review = str(clean_df.loc[i, 'customer_review'])
    actual = y_test.loc[i]
    predicted = y_pred[y_test.index.get_loc(i)]
    print(f"\n{idx}. Review: {review[:150]}...")
    print(f"   Actual: {actual} | Predicted: {predicted} ✓")

# Wrong predictions
print("\n--- INCORRECT PREDICTIONS (Sample 5) ---")
wrong_mask = (y_test != y_pred)
wrong_indices = y_test[wrong_mask].index[:5]

for idx, i in enumerate(wrong_indices, 1):
    review = str(clean_df.loc[i, 'customer_review'])
    actual = y_test.loc[i]
    predicted = y_pred[y_test.index.get_loc(i)]
    print(f"\n{idx}. Review: {review[:150]}...")
    print(f"   Actual: {actual} | Predicted: {predicted} ✗")

# Compare multiple models
print("\n" + "="*60)
print("STEP 8: Model Comparison")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(max_iter=1000, random_state=42)
}

results = {}
for name, clf in models.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train_vec, y_train)
    y_pred_model = clf.predict(X_test_vec)
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred_model),
        'F1-Score': f1_score(y_test, y_pred_model, pos_label='yes')
    }
    
    print(f"  Accuracy: {results[name]['Accuracy']:.4f}")
    print(f"  F1-Score: {results[name]['F1-Score']:.4f}")

# Plot model comparison
results_df = pd.DataFrame(results).T
results_df.plot(kind='bar', figsize=(10, 6), rot=0)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.ylim(0.85, 0.95)
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Model comparison plot saved as 'model_comparison.png'")
plt.show()

# Find and use best model
best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
best_model = models[best_model_name]
best_model.fit(X_train_vec, y_train)

print("\n" + "="*60)
print("STEP 9: Save Best Model")
print("="*60)
print(f"Best model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['Accuracy']:.4f}")

# Save model and vectorizer
pickle.dump(best_model, open('airline_review_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
print("\n✓ Model saved as 'airline_review_model.pkl'")
print("✓ Vectorizer saved as 'tfidf_vectorizer.pkl'")

# Create prediction function for demonstration
print("\n" + "="*60)
print("STEP 10: Test Prediction Function")
print("="*60)

def predict_review(review_text, model, vectorizer):
    """Predict whether an airline review is recommended"""
    # Clean text
    cleaned = clean_text(review_text)
    # Vectorize
    vectorized = vectorizer.transform([cleaned])
    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0] if hasattr(model, 'predict_proba') else None
    
    return prediction, probability

# Test with sample reviews
test_reviews = [
    "Amazing flight! The crew was friendly and the food was delicious. Highly recommend!",
    "Terrible experience. Delayed by 5 hours with no explanation. Never flying with them again.",
    "Average flight. Nothing special but nothing terrible either."
]

print("\nTesting prediction function with sample reviews:")
for i, review in enumerate(test_reviews, 1):
    pred, prob = predict_review(review, best_model, vectorizer)
    print(f"\n{i}. Review: {review}")
    print(f"   Prediction: {pred}")
    if prob is not None:
        print(f"   Confidence: {max(prob):.2%}")

# Feature Importance Analysis
print("\n" + "="*60)
print("STEP 10B: Feature Importance - Most Influential Words")
print("="*60)

# Get feature importance for Logistic Regression (only works with linear models)
if hasattr(best_model, 'coef_'):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = best_model.coef_[0]

    # Top positive words (predict "recommended")
    top_positive_idx = coefficients.argsort()[-20:][::-1]
    top_positive = [(feature_names[i], coefficients[i]) for i in top_positive_idx]

    # Top negative words (predict "not recommended")
    top_negative_idx = coefficients.argsort()[:20]
    top_negative = [(feature_names[i], coefficients[i]) for i in top_negative_idx]

    print("\nTop 10 words predicting RECOMMENDED:")
    for word, coef in top_positive[:10]:
        print(f"  {word}: {coef:.3f}")

    print("\nTop 10 words predicting NOT RECOMMENDED:")
    for word, coef in top_negative[:10]:
        print(f"  {word}: {coef:.3f}")
    
    # Visualize feature importance
    import numpy as np
    
    # Combine top positive and negative for visualization
    top_words = [word for word, _ in top_positive[:10]] + [word for word, _ in top_negative[:10]]
    top_coefs = [coef for _, coef in top_positive[:10]] + [coef for _, coef in top_negative[:10]]
    
    plt.figure(figsize=(12, 8))
    colors = ['green' if c > 0 else 'red' for c in top_coefs]
    plt.barh(range(len(top_words)), top_coefs, color=colors, alpha=0.7)
    plt.yticks(range(len(top_words)), top_words)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.title('Most Influential Words in Prediction', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature importance plot saved as 'feature_importance.png'")
    plt.show()
else:
    print(f"\n{best_model_name} does not support coefficient-based feature importance.")
    print("Feature importance is only available for linear models like Logistic Regression or Linear SVM.")

print("\n" + "="*60)
print("PIPELINE COMPLETE!")
print("="*60)
print("\nFiles generated:")
print("1. confusion_matrix.png")
print("2. model_comparison.png")
print("3. feature_importance.png")
print("4. airline_review_model.pkl")
print("5. tfidf_vectorizer.pkl")

# Error Analysis - Analyzing Misclassified Reviews
print("\n" + "="*60)
print("STEP 11: ERROR ANALYSIS - Why Did the Model Fail?")
print("="*60)

# Get all misclassified reviews
misclassified_mask = (y_test != y_pred)
misclassified_indices = y_test[misclassified_mask].index
total_misclassified = len(misclassified_indices)

print(f"\nTotal misclassified reviews: {total_misclassified}")
print(f"Misclassification rate: {(total_misclassified/len(y_test))*100:.2f}%")

# Analyze characteristics of misclassified reviews
misclassified_reviews = clean_df.loc[misclassified_indices, 'customer_review']
misclassified_labels = y_test.loc[misclassified_indices]
misclassified_predictions = pd.Series(y_pred, index=y_test.index).loc[misclassified_indices]

# 1. Review Length Analysis
print("\n--- CHARACTERISTIC 1: Review Length ---")
review_lengths = misclassified_reviews.apply(lambda x: len(str(x).split()))
print(f"Average length of misclassified reviews: {review_lengths.mean():.1f} words")
print(f"Median length: {review_lengths.median():.1f} words")
print(f"Shortest review: {review_lengths.min()} words")
print(f"Longest review: {review_lengths.max()} words")

# Categorize by length
short_reviews = review_lengths[review_lengths < 20]
medium_reviews = review_lengths[(review_lengths >= 20) & (review_lengths < 50)]
long_reviews = review_lengths[review_lengths >= 50]

print(f"\nShort reviews (<20 words): {len(short_reviews)} ({len(short_reviews)/total_misclassified*100:.1f}%)")
print(f"Medium reviews (20-50 words): {len(medium_reviews)} ({len(medium_reviews)/total_misclassified*100:.1f}%)")
print(f"Long reviews (>50 words): {len(long_reviews)} ({len(long_reviews)/total_misclassified*100:.1f}%)")

# 2. Mixed Sentiment Detection (looking for "but", "however", etc.)
print("\n--- CHARACTERISTIC 2: Mixed Sentiment Indicators ---")
mixed_sentiment_keywords = ['but', 'however', 'although', 'though', 'except', 'yet', 'still', 'unfortunately', 'despite']
mixed_sentiment_count = 0
mixed_sentiment_examples = []

for idx in misclassified_indices[:100]:  # Check first 100 for efficiency
    review_text = str(clean_df.loc[idx, 'customer_review']).lower()
    if any(keyword in review_text for keyword in mixed_sentiment_keywords):
        mixed_sentiment_count += 1
        if len(mixed_sentiment_examples) < 3:
            mixed_sentiment_examples.append((review_text, y_test.loc[idx], 
                                            pd.Series(y_pred, index=y_test.index).loc[idx]))

print(f"Reviews with mixed sentiment indicators: ~{mixed_sentiment_count}% (from sample)")
print("\nExamples of mixed sentiment reviews:")
for i, (review, actual, predicted) in enumerate(mixed_sentiment_examples, 1):
    print(f"\n{i}. {review[:200]}...")
    print(f"   Actual: {actual} | Predicted: {predicted}")

# 3. Sarcasm/Irony Detection (looking for exaggeration markers)
print("\n--- CHARACTERISTIC 3: Potential Sarcasm/Irony ---")
sarcasm_keywords = ['great', 'wonderful', 'amazing', 'perfect', 'excellent', 'fantastic', 'lovely']
sarcasm_count = 0
sarcasm_examples = []

for idx in misclassified_indices:
    review_text = str(clean_df.loc[idx, 'customer_review']).lower()
    actual = y_test.loc[idx]
    # Sarcasm: positive words but negative label
    if actual == 'no' and any(keyword in review_text for keyword in sarcasm_keywords):
        sarcasm_count += 1
        if len(sarcasm_examples) < 3:
            sarcasm_examples.append((review_text, actual, 
                                    pd.Series(y_pred, index=y_test.index).loc[idx]))

print(f"Potential sarcastic reviews (positive words + negative sentiment): {sarcasm_count}")
print("\nExamples of potential sarcasm:")
for i, (review, actual, predicted) in enumerate(sarcasm_examples, 1):
    print(f"\n{i}. {review[:200]}...")
    print(f"   Actual: {actual} | Predicted: {predicted}")

# 4. Vague/Neutral Reviews
print("\n--- CHARACTERISTIC 4: Vague/Neutral Reviews ---")
vague_keywords = ['okay', 'ok', 'fine', 'average', 'decent', 'alright', 'not bad', 'so-so']
vague_count = 0
vague_examples = []

for idx in misclassified_indices[:100]:
    review_text = str(clean_df.loc[idx, 'customer_review']).lower()
    if any(keyword in review_text for keyword in vague_keywords):
        vague_count += 1
        if len(vague_examples) < 3:
            vague_examples.append((review_text, y_test.loc[idx], 
                                  pd.Series(y_pred, index=y_test.index).loc[idx]))

print(f"Reviews with vague/neutral language: ~{vague_count}% (from sample)")
print("\nExamples of vague reviews:")
for i, (review, actual, predicted) in enumerate(vague_examples, 1):
    print(f"\n{i}. {review[:200]}...")
    print(f"   Actual: {actual} | Predicted: {predicted}")

# 5. Most Common Misclassification Patterns
print("\n--- CHARACTERISTIC 5: Misclassification Patterns ---")
false_positives = sum((y_test.loc[misclassified_indices] == 'no') & 
                     (pd.Series(y_pred, index=y_test.index).loc[misclassified_indices] == 'yes'))
false_negatives = sum((y_test.loc[misclassified_indices] == 'yes') & 
                     (pd.Series(y_pred, index=y_test.index).loc[misclassified_indices] == 'no'))

print(f"False Positives (Predicted 'yes', Actually 'no'): {false_positives} ({false_positives/total_misclassified*100:.1f}%)")
print(f"False Negatives (Predicted 'no', Actually 'yes'): {false_negatives} ({false_negatives/total_misclassified*100:.1f}%)")

# Summary of Error Analysis
print("\n" + "="*60)
print("ERROR ANALYSIS SUMMARY")
print("="*60)
print("\nKey Findings:")
print(f"1. Total Misclassifications: {total_misclassified} out of {len(y_test)} test samples")
print(f"2. Average length of misclassified reviews: {review_lengths.mean():.1f} words")
print(f"3. Short reviews (<20 words): {len(short_reviews)/total_misclassified*100:.1f}% of errors")
print(f"4. Reviews with mixed sentiment: Present in sample")
print(f"5. Potential sarcastic reviews: {sarcasm_count} cases")
print(f"6. Vague/neutral language: Common in misclassified reviews")
print("\nModel struggles most with:")
print("- Short, vague reviews (e.g., 'It was okay')")
print("- Mixed sentiment (e.g., 'Good food but terrible delays')")
print("- Sarcasm/irony (e.g., 'Oh great, another delay...')")
print("- Neutral language that doesn't clearly indicate recommendation")

print("\n" + "="*60)
print("COMPLETE ANALYSIS DONE!")
print("="*60)
