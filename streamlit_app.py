import streamlit as st
import pickle
import string

# --- Custom CSS for advanced theming ---
st.markdown("""
<style>
body, .main {background: #f6f8fb;}
.banner {
    background: linear-gradient(90deg,#05445E 70%,#189AB4 100%);
    color: white; margin:-48px -52px 30px -52px; padding:45px 10px 35px 10px; 
    border-radius:0 0 33px 33px; text-align:center;}
.smallcap {font-size:1.1em; color:#eee;}
.card {
    background:white; border-radius:14px; box-shadow:0 2px 8px rgba(30,70,140,0.110);
    padding:32px 34px 22px 34px; margin:20px 0;
}
.result-green {background:#3bb77e; color:white; font-size:1.25em; border-radius:9px; padding: 1.1em;}
.result-red {background:#f05254;color:white;font-size:1.25em; border-radius:9px; padding: 1.1em;}
.table th {background-color: #05445E;color:white;}
.metric-card {
    background:#F7F7F9; border-radius:14px; padding:16px;font-size:1.1em;
    box-shadow:0 1.5px 8px rgba(70,80,120,0.08);}
.metric-value {font-size:1.8em; font-weight:700; color:#05445E;}
.metric-label {font-size:1em; color:#606060;}
</style>
""", unsafe_allow_html=True)

# Banner
st.markdown("""
<div class='banner'>
    <h1>Airline Review Sentiment Analysis</h1>
    <div class='smallcap'>NLP Final Year Project &nbsp; | &nbsp; Logistic Regression + TF-IDF &nbsp; | &nbsp; 92% accuracy</div>
</div>
""", unsafe_allow_html=True)

# KPI metrics
col1, col2, col3 = st.columns(3)
with col1: st.markdown("<div class='metric-card'><span class='metric-value'>92%</span><div class='metric-label'>Accuracy</div></div>", unsafe_allow_html=True)
with col2: st.markdown("<div class='metric-card'><span class='metric-value'>0.92</span><div class='metric-label'>F1 Score</div></div>", unsafe_allow_html=True)
with col3: st.markdown("<div class='metric-card'><span class='metric-value'>64,440</span><div class='metric-label'>Reviews (Dataset)</div></div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True) # White card wrapper

# --- SINGLE REVIEW DEMO ---
st.header("Single Prediction Demo")
text = st.text_input("Review Text:")
if st.button("Predict"):
    if text.strip():
        @st.cache_resource
        def load_model():
            model = pickle.load(open('airline_review_model.pkl', 'rb'))
            vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
            return model, vectorizer
        def clean_text(t):
            t = str(t).lower()
            t = t.translate(str.maketrans('', '', string.punctuation + '0123456789'))
            return ' '.join(t.split())
        def predict_review(review, model, vectorizer):
            cleaned = clean_text(review)
            vect = vectorizer.transform([cleaned])
            pred = model.predict(vect)[0]
            prob = model.predict_proba(vect)[0]
            conf = max(prob)*100
            return pred, conf, prob
        model, vectorizer = load_model()
        pred, conf, prob = predict_review(text, model, vectorizer)
        result_class = 'result-green' if pred == "yes" else 'result-red'
        rec_text = 'Recommended' if pred == "yes" else 'Not Recommended'
        st.markdown(
            f"<div class='{result_class}'><b>{rec_text} &mdash; {conf:.1f}%</b></div>",
            unsafe_allow_html=True,
        )
        st.write(f"**Probabilities:**  Recommended: {prob[1]*100:.1f}%,  Not Recommended: {prob[0]*100:.1f}%")
    else:
        st.info("Please enter a review above.")
st.markdown("</div>", unsafe_allow_html=True)

# --- BATCH MODE ---

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.header("Batch Mode")
st.write("Paste multiple reviews below (one per line).")
batch_reviews = st.text_area("Batch Reviews:", height=100)
if st.button("Batch Analyze"):
    if batch_reviews.strip():
        batch_lines = [r.strip() for r in batch_reviews.split('\n') if r.strip()]
        model, vectorizer = load_model()
        rows = []
        rec_count = 0
        for i, line in enumerate(batch_lines, 1):
            pred, conf, _ = predict_review(line, model, vectorizer)
            rec_count += 1 if pred=="yes" else 0
            rows.append({
                "#": i, 
                "Review": line[:40] + ("..." if len(line) > 40 else ""),
                "Prediction": "Recommended" if pred=="yes" else "Not Recommended",
                "Confidence": f"{conf:.1f}%"
            })
        st.write(f"**Analysis:** {rec_count}/{len(rows)} reviews recommended ({rec_count/len(rows)*100:.1f}%)")
        st.table(rows)
    else:
        st.info("Paste some reviews above.")
st.markdown("</div>", unsafe_allow_html=True)

# --- AIRLINE COMPARISON ---

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.header("Airline Comparison Dashboard")
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
    comp_rows = []
    for airline, reviews in airlines.items():
        model, vectorizer = load_model()
        preds = [predict_review(r, model, vectorizer)[0]=="yes" for r in reviews]
        pct = sum(preds)/len(preds)*100
        stars = "⭐⭐⭐⭐⭐" if pct>=80 else "⭐⭐⭐⭐" if pct>=60 else "⭐⭐⭐" if pct>=40 else "⭐⭐"
        comp_rows.append({
            "Airline": airline,
            "Positive %": f"{pct:.0f}%",
            "Rating": stars
        })
    st.table(comp_rows)
st.markdown("</div>", unsafe_allow_html=True)

# --- Model Info ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
with st.expander("About this Project & Model", expanded=False):
    st.markdown("""
    **Project:** Airline Review Sentiment Analysis FYP  
    **Model:** Logistic Regression | Features: TF-IDF Vectorizer (5,000 words)  
    **Performance:** Accuracy 92% | F1-Score 0.92  
    **Dataset:** 64,440 airline reviews  
    **Author:** Your Name, University/Institution
    """)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.caption("© 2025 Your University | FYP Project Demo. All rights reserved.")
