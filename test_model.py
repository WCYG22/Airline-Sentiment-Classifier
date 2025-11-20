import pickle
import string

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation + '0123456789'))
    text = ' '.join(text.split())
    return text

# Load model and vectorizer
print("Loading model...")
model = pickle.load(open('airline_review_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
print("✓ Loaded successfully!\n")

# Test predictions
reviews = [
    "Excellent service and comfortable flight!",
    "Terrible delays and rude staff.",
    "Average experience, nothing special."
]

for review in reviews:
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(vectorized)[0]
        confidence = max(prob) * 100
        print(f"Review: {review}")
        print(f"Prediction: {prediction} ({confidence:.1f}% confidence)\n")
    else:
        print(f"Review: {review}")
        print(f"Prediction: {prediction}\n")
