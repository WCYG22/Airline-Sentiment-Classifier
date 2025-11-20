import streamlit as st
import pickle
import string

st.set_page_config(
    page_title="Airline Review NLP Sentiment Analysis",
    page_icon="✈️",
    layout="wide"
)

# Optional: Add your logo or diagram here
# st.image("your_nlp_model_architecture.png", width=350)
st.title("Airline Review Sentiment Analysis")
st.subheader("A Professional NLP Demo Application")
st.caption("Powered by Logistic Regression + TF-IDF • 92% accuracy")

# Project summary at the top
with st.expander("[+] Project Overview", expanded=True):
    st.write(
        "This demo uses an NLP sentiment classifier to analyze airline reviews. "
        "It predicts recommendation status ('Recommended' or 'Not Recommended') "
        "with probability confidence, shows batch statistics, and benchmarks multiple airlines."
    )

# Utility functions
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation + '0123456789'))
    return ' '.join(text.split())

@st.cache_resource
def load_model():
    model = pickle.load(open("airline_review_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0]
    conf = max(prob) * 100
    return pred, conf, prob

model, vectorizer = load_model()

# --- SINGLE PREDICTION --- #
st.markdown("### Single Review Analysis")
with st.form("single_review_form", clear_on_submit=False):
    review = st.text_area("Enter an airline review", height=100)
    submit = st.form_submit_button("Predict Sentiment")
if submit and review.strip():
    pred, conf, prob = predict(review, model, vectorizer)
    color = "green" if pred == "yes" else "red"
    st.markdown(
        f"<div style='background-color: {color}; color: white; padding: 12px; border-radius: 5px;'>"
        f"<b>Prediction:</b> {'Recommended' if pred == 'yes' else 'Not Recommended'} "
        f"(<b>{conf:.1f}%</b> confidence)<br>"
        f"<small>Probability: Recommended {prob[1]*100:.1f}% | Not Recommended {prob[0]*100:.1f}%</small></div>",
        unsafe_allow_html=True,
    )
else:
    st.caption("Enter a review and click 'Predict Sentiment'.")

st.divider()

# --- BATCH PREDICTION --- #
st.markdown("### Batch Analysis")
col1, col2 = st.columns([2, 1])
with col1:
    multi_reviews = st.text_area("Paste multiple reviews (one per line)", height=100, key="batch_reviews")
with col2:
    run_batch = st.button("Run Batch Analysis", key="run_batch")

if run_batch and multi_reviews.strip():
    reviews = [r.strip() for r in multi_reviews.split("\n") if r.strip()]
    results = [predict(r, model, vectorizer) for r in reviews]
    num_rec = sum(1 for p,c,pr in results if p == "yes")
    avg_conf = sum(c for p,c,pr in results) / len(results)
    st.success(
        f"{num_rec} of {len(results)} reviews recommended "
        f"({num_rec/len(results)*100:.1f}%). Avg confidence: {avg_conf:.1f}%"
    )
    st.markdown("#### Details")
    for i, (rev, (p,c,pr)) in enumerate(zip(reviews, results), 1):
        label = "Recommended" if p == "yes" else "Not Recommended"
        col = "green" if p == "yes" else "red"
        st.markdown(
            f"<b>{i}. <span style='color:{col}'>{label}</span></b> ({c:.1f}%)<br>"
            f"<span style='font-size:0.92em; color:#555'>{rev[:80]}{'...' if len(rev)>80 else ''}</span>",
            unsafe_allow_html=True
        )

st.divider()

# --- AIRLINE COMPARISON --- #
with st.expander("Airline Comparison Dashboard", expanded=False):
    st.markdown("Analyze sentiment for different airlines (preloaded sample).")
    if st.button("Compare Airlines"):
        airlines = {
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
        for name, reviews in airlines.items():
            preds = [predict(r, model, vectorizer)[0]=="yes" for r in reviews]
            confs = [predict(r, model, vectorizer)[1] for r in reviews]
            pct = sum(preds)/len(preds)*100
            avg_c = sum(confs)/len(confs)
            rating = "⭐⭐⭐⭐⭐" if pct>=80 else "⭐⭐⭐⭐" if pct>=60 else "⭐⭐⭐" if pct>=40 else "⭐⭐"
            st.info(f"{name}: {rating} | Positive: {pct:.0f}% | Confidence: {avg_c:.1f}%")

# --- Model Details Section ---

with st.expander("Show Model Details"):
    st.code("""
Model: Logistic Regression
Vectorizer: TF-IDF (5,000 words)
Training Accuracy: 92%
Precision, Recall, F1: 0.92
Dataset: 64,440 airline reviews
""", language="text")

st.markdown("""
<style>
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
st.write("")


