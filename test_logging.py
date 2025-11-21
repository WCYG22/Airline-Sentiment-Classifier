"""Test script to verify the alert logging functionality"""
import pickle
from datetime import datetime

# Load model
model = pickle.load(open('airline_review_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Test review
test_review = "Terrible service, rude staff, never flying again"

# Clean and predict
import string
cleaned = test_review.lower().translate(str.maketrans('', '', string.punctuation + '0123456789'))
vect = vectorizer.transform([cleaned])
pred = model.predict(vect)[0]
prob = model.predict_proba(vect)[0]
conf = max(prob) * 100

print(f"Review: {test_review}")
print(f"Prediction: {pred}")
print(f"Confidence: {conf:.1f}%")
print()

# Test logging
if pred == "no" and conf >= 85:
    print("✅ Alert condition met - attempting to log...")
    try:
        with open('alert_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"ALERT FLAGGED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Review: {test_review}\n")
            f.write(f"Prediction: NOT RECOMMENDED\n")
            f.write(f"Confidence: {conf:.1f}%\n")
            f.write(f"{'='*70}\n")
        print("✅ Successfully wrote to alert_log.txt")
        
        # Read back to verify
        with open('alert_log.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"\n📄 Last 10 lines of alert_log.txt:")
            print(''.join(lines[-10:]))
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"❌ Alert condition NOT met (pred={pred}, conf={conf:.1f}%)")
