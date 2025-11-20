import streamlit as st
import pickle
import string

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {background-color: #f4f5fa;}
    .small-label { color: #666; font-size: 0.92em;}
    .highlight-card {
        background: linear-gradient(90deg,#0288d1 60%,#26c6da 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 18px;
        box-shadow: 1px 2px 5px rgba(40,80,160,0.08);
    }
    .section {
        background: white; 
        border-radius: 10px; 
        box-shadow: 1px 2px 6px rgba(80,80,80,0.05);
        margin: 32px 0 18px 0;
        padding: 30px 38px 22px 38px;
    }
    th {background: #0288d1; color: white;}
    .banner {
        padding: 20px 0 10px 0;
        margin-bottom: 16px;
        border-radius: 0px 0px 18px 18px;
        background: linear-gradient(90deg,#0288d1,#26c6da);
        color: white;
        text-align: center;
        box-shadow: 1px 4px 14px rgba(32,64,120,0.09);
    }
    </style>
""", unsafe_allow_html=True)

# --- Banner ---
st.markdown(
    "<div class='banner'><h1>Airline Review Sentiment Analysis</h1>"
    "<h3 class='small-label'>NLP FYP Project &mdash; Logistic Regression + TF-IDF (92% accuracy)</h3></div>", 
    unsafe_allow_html=True
)

# --- Sidebar (minimal) ---
st.sidebar.title("Navigation")
st.sidebar.info("This is a demo for NLP airline review sentiment classification.")
st.sidebar.write("Jump to:")
st.sidebar.markdown("- [Single Prediction](#single-prediction-demo)")
st.sidebar.markdown("- [Batch Analysis](#batch-mode)")
st.sidebar.markdown("- [Airline Comparison](#airline-comparison-dashboard)")
st.sidebar.markdown("- [Model Information](#model-information)")
st.sidebar.markdown("---")
st.sidebar.write("Developed by: Your Name")

# --- Model Functions ---
@st.cache_resource
def load_model():
    model = pickle.load(open('airline_review_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    return model, vectorizer
def clean_text(t):
    t = str(t).lower()
    t = t.translate(str.maketrans('', '', string.punctuation + '0123456789'))
    return ' '.join(t.split())
def predict_review(r, m, v):
    cleaned = clean_text(r)
    vect = v.transform([cleaned])
    pred = m.predict(vect)[0]
    prob = m.predict_proba(vect)[0]
    conf = max(prob)*100
    return pred, conf, prob

model, vectorizer = load_model()

# --- SINGLE PREDICTION DEMO ---

st.markdown("<div class='section' id='single-prediction-demo'>", unsafe_allow_html=True)
st.header("Single Prediction Demo")
user_review = st.text_input("Enter an airline review for instant prediction", "")
if st.button("Predict Now"):
    if user_review.strip():
        pred, conf, prob = predict_review(user_review, model, vectorizer)
        card_color = "#2e7d32" if pred == "yes" else "#d32f2f"
        icon = "✅" if pred == "yes" else "❌"
        st.markdown(
            f"<div class='highlight-card' style='background: {card_color};'>"
            f"<span style='font-size:2.0em'>{icon}</span>  <b>Prediction:</b> {'Recommended' if pred=='yes' else 'Not Recommended'}<br>"
            f"<span class='small-label'>Confidence: <b>{conf:.1f}%</b> | "
            f"Probabilities: Rec {prob[1]*100:.1f}%, Not {prob[0]*100:.1f}%</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Please enter a review to predict.")

st.markdown("</div>", unsafe_allow_html=True)

# --- BATCH MODE ---
st.markdown("<div class='section' id='batch-mode'>", unsafe_allow_html=True)
st.header("Batch Mode")
st.write("Paste multiple reviews (one per line) to analyze all at once.")
batch_text = st.text_area("Example: Review1 \n Review2...", height=100, key="batchmode")
if st.button("Batch Analyze"):
    if batch_text.strip():
        lines = [r.strip() for r in batch_text.split('\n') if r.strip()]
        results = [predict_review(r, model, vectorizer) for r in lines]
        num_rec = sum(1 for p,_,_ in results if p == "yes")
        table_data = []
        for i, (r, (pred, conf, _)) in enumerate(zip(lines, results), 1):
            table_data.append({"#": i, "Review": r[:40] + ("..." if len(r)>40 else ""), 
                            "Pred": "Recommended" if pred=="yes" else "Not Rec.", 
                            "Conf": f"{conf:.1f}%"})
        st.markdown("<b>Summary Table:</b>", unsafe_allow_html=True)
        st.table(table_data)
        st.success(f"{num_rec} out of {len(results)} reviews are recommended. ({num_rec/len(results)*100:.1f}%)")
    else:
        st.info("Paste reviews above and click Batch Analyze.")
st.markdown("</div>", unsafe_allow_html=True)

# --- AIRLINE COMPARISON DASHBOARD ---
st.markdown("<div class='section' id='airline-comparison-dashboard'>", unsafe_allow_html=True)
st.header("Airline Comparison Dashboard")
st.write("Compare pre-loaded airlines on sentiment metrics using sample reviews.")
if st.button("Run Airline Comparison"):
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
    out = []
    for name, reviews in airlines.items():
        preds = [predict_review(r, model, vectorizer)[0]=="yes" for r in reviews]
        confs = [predict_review(r, model, vectorizer)[1] for r in reviews]
        pct = sum(preds)/len(preds)*100
        avg_c = sum(confs)/len(confs)
        rating = "⭐⭐⭐⭐⭐" if pct>=80 else "⭐⭐⭐⭐" if pct>=60 else "⭐⭐⭐" if pct>=40 else "⭐⭐"
        out.append({
            "Airline": name,
            "Positive %": f"{pct:.0f}%",
            "Avg Conf": f"{avg_c:.1f}%",
            "Rating": rating
        })
    st.markdown("<b>Comparison Table:</b>", unsafe_allow_html=True)
    st.table(out)
st.markdown("</div>", unsafe_allow_html=True)

# --- Model Information ---
st.markdown("<div class='section' id='model-information'>", unsafe_allow_html=True)
st.header("Model Information")
info_table = [
    ["Model", "Logistic Regression"],
    ["Vectorizer", "TF-IDF (5,000 words)"],
    ["Training Accuracy", "92%"],
    ["Precision / Recall / F1", "0.92 (each)"],
    ["Dataset", "64,440 airline reviews"],
]
st.table(info_table)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("© Your University | FYP NLP Demo | All rights reserved")
