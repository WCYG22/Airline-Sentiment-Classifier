import streamlit as st
import pickle
import string

# --- Sidebar information ---
with st.sidebar:
    st.header("Model Information")
    st.write("Model: Logistic Regression")
    st.write("Features: TF-IDF vectorizer (5,000 words)")
    st.write("Accuracy: 92%")
    st.write("Dataset: 64,440 airline reviews")
    st.markdown("---")
    st.header("About")
    st.write("Sentiment analysis of airline reviews.\nPredicts 'recommended' or 'not recommended' with confidence.")

# --- Utility functions ---
@st.cache_resource
def load_model():
    model = pickle.load(open('airline_review_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    return model, vectorizer

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation + '0123456789'))
    return ' '.join(text.split())

def predict_review(review_text, model, vectorizer):
    cleaned = clean_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]
    confidence = max(prob) * 100
    return prediction, confidence, prob

model, vectorizer = load_model()

st.title("Airline Review Sentiment Classifier")
st.write("Enter one or more airline reviews to predict recommendation sentiment.")

# --- SINGLE REVIEW SECTION ---
st.subheader("Single Review Prediction")
single_review = st.text_area("Enter a review for prediction", key="single_review", height=100)
if st.button("Analyze Review"):
    if single_review.strip():
        pred, conf, prob = predict_review(single_review, model, vectorizer)
        st.markdown("**Result:** " + ("Recommended" if pred == "yes" else "Not Recommended"))
        st.markdown(f"**Confidence:** {conf:.1f}%")
        st.progress(int(conf))
        st.markdown(f"Probability - Recommended: {prob[1]*100:.1f}%, Not Recommended: {prob[0]*100:.1f}%")
        if pred == "no" and conf >= 85:
            st.warning("High confidence negative review detected. Customer service follow-up recommended.")
    else:
        st.info("Please enter a review.")

# --- BATCH ANALYSIS SECTION ---
with st.expander("Batch Review Analysis (Multiple Reviews)", expanded=False):
    batch_input = st.text_area("Enter each review on a new line", key="batch_input", height=120)
    if st.button("Analyze Batch"):
        if batch_input.strip():
            reviews = [r.strip() for r in batch_input.split('\n') if r.strip()]
            summary = {"rec": 0, "not": 0, "conf_total": 0}
            results = []
            for review in reviews:
                pred, conf, prob = predict_review(review, model, vectorizer)
                results.append((review, pred, conf))
                summary["conf_total"] += conf
                if pred == "yes":
                    summary["rec"] += 1
                else:
                    summary["not"] += 1
            avg_conf = summary["conf_total"] / len(results)
            st.markdown(f"**Total:** {len(results)} | "
                        f"Recommended: {summary['rec']} ({summary['rec']/len(results)*100:.1f}%) | "
                        f"Not Recommended: {summary['not']} ({summary['not']/len(results)*100:.1f}%) | "
                        f"Avg Confidence: {avg_conf:.1f}%")
            st.markdown("---")
            for i, (review, pred, conf) in enumerate(results, 1):
                st.write(f"{i}. {'Recommended' if pred == 'yes' else 'Not Recommended'} ({conf:.1f}%)")
                with st.expander(f"Show Review {i}"):
                    st.write(review)
        else:
            st.info("Please enter at least one review.")

# --- AIRLINE COMPARISON SECTION ---
with st.expander("Airline Comparison Dashboard", expanded=False):
    st.write("Analyze sample reviews across multiple airlines (pre-loaded demo).")
    if st.button("Run Comparison"):
        airlines_data = {
            "Singapore Airlines": [
                "Excellent service and comfortable seats throughout the flight",
                "Best airline I've ever flown with, highly professional crew",
                "Great experience from booking to landing, will fly again",
                "Comfortable journey with top-notch entertainment system"
            ],
            "Budget Airways": [
                "Terrible service with constant delays and no communication",
                "Uncomfortable cramped seats and rude cabin crew",
                "Lost my baggage and received no help from staff",
                "Worst flying experience, avoid at all costs"
            ],
            "National Carrier": [
                "Good flight overall but nothing exceptional",
                "Average service, reasonable price, got me there safely",
                "Decent experience, some delays but acceptable",
                "Okay for the price, nothing to complain about"
            ]
        }
        for airline, reviews in airlines_data.items():
            metrics = []
            for review in reviews:
                pred, conf, _ = predict_review(review, model, vectorizer)
                metrics.append((pred, conf))
            rec_count = sum(1 for p, _ in metrics if p == "yes")
            avg_conf = sum(c for _, c in metrics) / len(metrics)
            pos_pct = rec_count / len(metrics) * 100
            rating = "⭐" * (5 if pos_pct >= 80 else 4 if pos_pct >= 60 else 3 if pos_pct >= 40 else 2)
            st.markdown(f"**{airline}**")
            st.write(f"Positive Sentiment: {pos_pct:.1f}% | Avg Confidence: {avg_conf:.1f}% | Reviews: {len(metrics)}")
            st.write(f"Overall Rating: {rating}")
            st.markdown("---")

st.markdown("---")
st.caption("Demo by Airline Review Sentiment Classifier | Powered by Streamlit")
