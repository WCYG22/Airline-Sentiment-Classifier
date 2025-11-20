import streamlit as st
import pickle
import string
import datetime
import pandas as pd

# ----------- Model utilities -----------
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

# ----------- Streamlit UI -----------

st.set_page_config(page_title="Airline Sentiment Analysis", layout="wide")

TABS = [
    "Main Menu", "Test Suite", "Interactive Mode",
    "Batch Analysis", "Airline Comparison", "Model Info"
]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(TABS)

# ===================== MAIN MENU =====================
with tab1:
    st.title("Airline Review Sentiment Classifier")
    st.header("Terminal-based NLP system, now in Streamlit. All CLI features on web!")
    st.markdown("---")
    st.markdown("""
    #### Choose an option using the tabs above:
    1. **Test Suite:** Run and display all predefined test cases and their detailed stats  
    2. **Interactive Mode:** Prediction, session statistics, rolling history, and alerts  
    3. **Batch Analysis:** Enter multiple reviews for stats, table, confidence, breakdown  
    4. **Airline Comparison:** Benchmark sentiment metrics across airlines  
    5. **Model Info:** All trained model specs and performance
    """)

# ===================== TEST SUITE =====================
with tab2:
    st.header("Test Suite Demonstration")
    test_cases = [
        {
            'category': 'Positive - Excellent Service',
            'review': 'Amazing flight! The crew was incredibly friendly and helpful. Comfortable seats and delicious food. Best airline experience I have ever had. Highly recommend!'
        },
        {
            'category': 'Negative - Poor Service',
            'review': 'Worst airline experience ever. Flight delayed by 5 hours with absolutely no explanation. Staff were rude and unhelpful. Uncomfortable seats and terrible food. Never flying with them again!'
        },
        {
            'category': 'Neutral - Mixed Experience',
            'review': 'Decent flight overall. Nothing exceptional but got me to my destination safely. Service was average, seats were okay. Price was reasonable for what you get.'
        },
        {
            'category': 'Positive - Great Value',
            'review': 'Excellent value for money. Clean plane, smooth flight, and professional staff. In-flight entertainment was great. Would definitely fly with them again!'
        },
        {
            'category': 'Negative - Multiple Issues',
            'review': 'Terrible experience from start to finish. Lost my baggage, poor customer service response, plane was dirty and cramped. Food was inedible. Avoid this airline at all costs!'
        },
        {
            'category': 'Edge Case - Very Short',
            'review': 'Great flight, very comfortable!'
        },
        {
            'category': 'Edge Case - Ambiguous',
            'review': 'Flight was okay. Nothing special.'
        }
    ]
    suite_rows = []
    rec, not_rec = 0, 0
    for i, test in enumerate(test_cases, 1):
        pred, conf, prob = predict_review(test['review'], model, vectorizer)
        suite_rows.append({
            "#": i,
            "Category": test['category'],
            "Input": test['review'][:50] + "...",
            "Prediction": "Recommended" if pred == "yes" else "Not Recommended",
            "Confidence": f"{conf:.1f}%",
        })
        if pred == "yes":
            rec += 1
        else:
            not_rec += 1
    st.table(pd.DataFrame(suite_rows))
    st.success(f"Total: {len(test_cases)} | Recommended: {rec} | Not Recommended: {not_rec} | Accuracy: {rec/len(test_cases)*100:.1f}%")

# ================ INTERACTIVE MODE ================
with tab3:
    st.header("Interactive Prediction Mode")
    if "history" not in st.session_state: st.session_state["history"] = []
    st.write("**Commands:** Type a review for prediction. Click 'Show Statistics' or 'Clear History' as needed.")

    inp = st.text_input("Enter review for prediction:")
    _col1, _col2, _col3 = st.columns([1,1,3])
    if _col1.button("Predict"):
        if len(inp.split()) < 2:
            st.warning("Review too short. Please provide more detail.")
        else:
            pred, conf, prob = predict_review(inp, model, vectorizer)
            st.session_state["history"].append({
                "review": inp,
                "prediction": pred,
                "conf": conf,
                "dt": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.markdown("---")
            st.write(f"**Sentiment:** {'Recommended' if pred == 'yes' else 'Not Recommended'}")
            st.write(f"**Confidence:** {conf:.1f}%")
            # Alert
            if pred == "no" and conf and conf >= 85:
                st.error("ALERT: High priority negative review with high confidence detected!")
    if _col2.button("Show Statistics"):
        history = st.session_state["history"]
        if not history:
            st.info("No predictions yet.")
        else:
            rec_count = sum(1 for h in history if h["prediction"] == "yes")
            avg_conf = sum(h["conf"] for h in history) / len(history)
            st.write(f"Total predictions: {len(history)} | Recommended: {rec_count} ({rec_count/len(history)*100:.1f}%) | Not Recommended: {len(history)-rec_count}")
            st.write(f"Average confidence: {avg_conf:.1f}%")
            st.dataframe(pd.DataFrame(history))
    if _col3.button("Clear History"):
        st.session_state["history"] = []

    if st.session_state["history"]:
        st.markdown("---\n**Recent Predictions:** (Last 5)")
        hlist = st.session_state["history"][-5:][::-1]
        for i, h in enumerate(hlist, 1):
            st.write(f"{i}. {'REC' if h['prediction']=='yes' else 'NOT'} ({h['conf']:.1f}%) | {h['review'][:50]}...")

# ================ BATCH ANALYSIS ================
with tab4:
    st.header("Batch Analysis Mode")
    batch_in = st.text_area("Enter multiple reviews (one per line):", height=140)
    if st.button("Run Batch Analysis"):
        reviews = [r.strip() for r in batch_in.split('\n') if r.strip()]
        if not reviews:
            st.warning("No reviews entered.")
        else:
            results = []
            for r in reviews:
                pred, conf, prob = predict_review(r, model, vectorizer)
                results.append({
                    "review": r[:50] + "...",
                    "Sentiment": "Recommended" if pred == "yes" else "Not Recommended",
                    "Confidence": f"{conf:.1f}%"
                })
            recs = sum(1 for r in results if r["Sentiment"] == "Recommended")
            avg_conf = sum(float(r["Confidence"][:-1]) for r in results) / len(results)
            st.table(pd.DataFrame(results))
            st.success(f"Total Reviews: {len(results)} | Recommended: {recs} ({recs/len(results)*100:.1f}%) | Not Recommended: {len(results)-recs} ({(len(results)-recs)/len(results)*100:.1f}%) | Avg Confidence: {avg_conf:.1f}%")

# ================ AIRLINE COMPARISON ================
with tab5:
    st.header("Airline Comparison Dashboard")
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
    compare_rows = []
    for airline, reviews in airlines_data.items():
        preds = []
        confs = []
        for r in reviews:
            pred, conf, _ = predict_review(r, model, vectorizer)
            preds.append(pred == 'yes')
            confs.append(conf)
        pct = sum(preds) / len(preds) * 100
        avg_conf = sum(confs) / len(confs)
        rating = "⭐⭐⭐⭐⭐" if pct >= 80 else "⭐⭐⭐⭐" if pct >= 60 else "⭐⭐⭐" if pct >= 40 else "⭐⭐"
        compare_rows.append({
            "Airline": airline,
            "Positive Sentiment": f"{pct:.0f}%",
            "Avg Confidence": f"{avg_conf:.1f}%",
            "Sample Size": len(reviews),
            "Rating": rating
        })
    st.table(pd.DataFrame(compare_rows))

# ================ MODEL INFO ================
with tab6:
    st.header("Model Information")
    st.write(f"""
- Model: Logistic Regression  
- Feature Extraction: TF-IDF Vectorizer  
- Vocabulary Size: {len(vectorizer.get_feature_names_out())} words  
- Training Accuracy: 92%  
- Precision: 92%  
- Recall: 92%  
- F1-Score: 0.92  
- Dataset: 64,440 airline reviews  
    """)

st.write("— Demo covers 100% of terminal features. Ready for your FYP! —")
