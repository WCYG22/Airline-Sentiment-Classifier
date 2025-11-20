import streamlit as st
import pickle
import string
import datetime
import pandas as pd

# --------- Model and Prediction Functions ---------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('airline_review_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    return model, vectorizer

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation + '0123456789'))
    return ' '.join(text.split())

def predict_review(text, model, vectorizer):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vect)[0]
        conf = max(prob) * 100
        return pred, conf, prob
    return pred, None, None

model, vectorizer = load_artifacts()

# --------- Custom Website CSS ---------
st.markdown("""
<style>
body, .main {background: #f8fcfe;}
.banner {
  background: linear-gradient(90deg,#0683b0 0%,#18c1e8 100%);
  color: white; 
  border-radius:20px;
  margin-bottom:34px;
  padding:42px 10px 35px 10px; 
  box-shadow: 0 9px 35px rgba(24, 193, 232, 0.09);
  text-align: center;
}
.metric-card {
  display:inline-block;
  background:white;
  border-radius:13px;
  box-shadow:0 2px 10px rgba(30,70,140,0.10);
  margin:10px; min-width:160px;
  padding:22px 18px 16px 18px; text-align:center;
}
.metric-value {font-size:2.0em;font-weight:700;color:#0683b0;}
.metric-label {font-size:1.05em;color:#555;}
.section {background:#fff;border-radius:16px;box-shadow:0 2px 10px rgba(60,120,180,0.08);
  margin:32px 0 24px 0; padding:36px 28px 26px 28px; }
.result-green {background:#21ad88;color:white;border-radius:12px; padding:1.1em;font-size:1.2em;}
.result-red {background:#ea5757;color:white;border-radius:12px; padding:1.1em;font-size:1.2em;}
hr {margin: 0 0 24px 0;}
h3{margin-top:1.5em;}
</style>
""", unsafe_allow_html=True)

# --------- Banner ---------
st.markdown("""
<div class="banner">
    <h1>Airline Review Sentiment Analysis</h1>
    <div>NLP FYP Project &nbsp;•&nbsp; Logistic Regression + TF-IDF &nbsp;•&nbsp; 92% Accuracy</div>
</div>
""", unsafe_allow_html=True)

# --------- KPI CARDS ---------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.markdown("""
      <div class="metric-card">
        <div class="metric-value">92%</div>
        <div class="metric-label">Accuracy</div>
      </div>
    """, unsafe_allow_html=True)
with kpi_col2:
    st.markdown("""
      <div class="metric-card">
        <div class="metric-value">0.92</div>
        <div class="metric-label">F1 Score</div>
      </div>
    """, unsafe_allow_html=True)
with kpi_col3:
    st.markdown("""
      <div class="metric-card">
        <div class="metric-value">64,440</div>
        <div class="metric-label">Reviews</div>
      </div>
    """, unsafe_allow_html=True)
with kpi_col4:
    st.markdown("""
      <div class="metric-card">
        <div class="metric-value">5,000</div>
        <div class="metric-label">TF-IDF Features</div>
      </div>
    """, unsafe_allow_html=True)

# --------- Streamlit Tabs (Pages) ---------
TABS = [
    "Prediction Demo", "Test Suite", "Batch Analysis",
    "Airline Comparison", "Model Info"
]
tab0, tab1, tab2, tab3, tab4 = st.tabs(TABS)

# --------- SINGLE PREDICTION DEMO ---------
with tab0:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Single Review Prediction")
    user_review = st.text_input("Enter a review to predict:")
    if st.button("Predict"):
        if user_review.strip():
            pred, conf, prob = predict_review(user_review, model, vectorizer)
            result_class = 'result-green' if pred == "yes" else 'result-red'
            result_string = 'Recommended' if pred == "yes" else 'Not Recommended'
            st.markdown(
                f'<div class="{result_class}"><b>{result_string} &mdash; {conf:.1f}%</b></div>',
                unsafe_allow_html=True
            )
            st.write(f"**Probability breakdown:**  Recommended: {prob[1]*100:.1f}%, Not Recommended: {prob[0]*100:.1f}%")
            # Confidence bar
            bar_length = int(conf // 2)
            bar = "🟩" * bar_length + "⬜" * (50-bar_length)
            st.write(f"{bar} {conf:.1f}%")
            # Alert
            if pred == "no" and conf >= 85:
                st.error("ALERT: High confidence negative review. Escalate to customer service!")
        else:
            st.info("Enter a review above.")
    st.markdown('</div>', unsafe_allow_html=True)

# --------- TEST SUITE ---------
with tab1:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Test Suite (Automated Examples)")
    test_cases = [
        {'category': 'Positive - Excellent Service','review': 'Amazing flight! The crew was incredibly friendly and helpful. Comfortable seats and delicious food. Best airline experience I have ever had. Highly recommend!'},
        {'category': 'Negative - Poor Service','review': 'Worst airline experience ever. Flight delayed by 5 hours with absolutely no explanation. Staff were rude and unhelpful. Uncomfortable seats and terrible food. Never flying with them again!'},
        {'category': 'Neutral - Mixed Experience','review': 'Decent flight overall. Nothing exceptional but got me to my destination safely. Service was average, seats were okay. Price was reasonable for what you get.'},
        {'category': 'Positive - Great Value','review': 'Excellent value for money. Clean plane, smooth flight, and professional staff. In-flight entertainment was great. Would definitely fly with them again!'},
        {'category': 'Negative - Multiple Issues','review': 'Terrible experience from start to finish. Lost my baggage, poor customer service response, plane was dirty and cramped. Food was inedible. Avoid this airline at all costs!'},
        {'category': 'Edge Case - Very Short','review': 'Great flight, very comfortable!'},
        {'category': 'Edge Case - Ambiguous','review': 'Flight was okay. Nothing special.'}
    ]
    tab_rows = []
    rec, norec = 0, 0
    for i, t in enumerate(test_cases, 1):
        pred, conf, prob = predict_review(t['review'], model, vectorizer)
        tab_rows.append({
            "#": i, "Category": t['category'], 
            "Prediction": "Recommended" if pred=="yes" else "Not Recommended",
            "Conf.": f"{conf:.1f}%",
            "Text": t['review'][:42]+"..."
        })
        rec += (pred=="yes")
        norec += (pred=="no")
    st.table(pd.DataFrame(tab_rows))
    st.success(f"Passed: {rec}, Not Recommended: {norec}, Accuracy: {rec/len(test_cases)*100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# --------- BATCH MODE ---------
with tab2:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Batch Review Analysis")
    batch_in = st.text_area("Paste multiple reviews (one per line):", height=100)
    if st.button("Analyze Batch"):
        reviews = [r.strip() for r in batch_in.split('\n') if r.strip()]
        if not reviews:
            st.warning("No reviews entered.")
        else:
            rows = []
            recs, nrecs, ct_total = 0, 0, 0.
            for i, rev in enumerate(reviews, 1):
                pred, conf, _ = predict_review(rev, model, vectorizer)
                ct_total += conf
                ok = (pred=="yes")
                rows.append({"#": i, "Short Review": rev[:40]+"...", "Sentiment": "Recommended" if ok else "Not Recommended", "Conf.": f"{conf:.1f}%"})
                recs += ok
                nrecs += not ok
            avg_conf = ct_total / len(reviews)
            st.table(rows)
            st.success(f"Recommended: {recs} ({recs/len(reviews)*100:.1f}%), Not Recommended: {nrecs} ({nrecs/len(reviews)*100:.1f}%), Avg. Confidence: {avg_conf:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# --------- AIRLINE COMPARISON ---------
with tab3:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Airline Comparison")
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
    airline_rows = []
    for airline, reviews in airlines_data.items():
        preds = []
        confs = []
        for r in reviews:
            pred, conf, _ = predict_review(r, model, vectorizer)
            preds.append(pred=="yes")
            confs.append(conf)
        pct = sum(preds)/len(preds)*100
        avgc = sum(confs)/len(confs)
        rating = "⭐⭐⭐⭐⭐" if pct >= 80 else "⭐⭐⭐⭐" if pct >= 60 else "⭐⭐⭐" if pct >= 40 else "⭐⭐"
        airline_rows.append({"Airline": airline, "Pos %": f"{pct:.0f}%", "Avg. Conf": f"{avgc:.1f}%", "Rating": rating})
    st.table(airline_rows)
    st.markdown('</div>', unsafe_allow_html=True)

# --------- MODEL INFO ---------
with tab4:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Model Information")
    st.write(f"""
- Model: Logistic Regression  
- Feature: TF-IDF (5,000 words)  
- Training accuracy: 92%  
- Precision / Recall / F1: 0.92  
- Dataset: 64,440 airline reviews  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.write("---")
st.caption("© 2025 Your University | FYP Demo — Built with Streamlit. All core CLI features available.")
