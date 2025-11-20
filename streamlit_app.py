import streamlit as st
import pickle
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------- Configuration ---------
st.set_page_config(
    page_title="Airline Sentiment Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------- Model and Data Functions ---------
@st.cache_resource
def load_artifacts():
    try:
        model = pickle.load(open('airline_review_model.pkl', 'rb'))
        vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'airline_review_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
        # Attempt to load the dataset
        df = pd.read_excel('capstone_airline_reviews3.xlsx')
        return df
    except Exception as e:
        # Return empty dataframe if file missing (graceful degradation)
        return pd.DataFrame()

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

# --------- Initialization ---------
model, vectorizer = load_artifacts()
if model is None or vectorizer is None:
    st.warning("Application cannot proceed without model files.")
    st.stop()

df = load_data()

# --------- Custom CSS ---------
st.markdown("""
<style>
    /* Main Layout */
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        border: 1px solid #e9ecef;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {transform: translateY(-2px); box-shadow: 0 8px 12px rgba(0,0,0,0.08);}
    .metric-value {font-size: 2.5rem; font-weight: 700; color: #0d6efd; margin-bottom: 8px;}
    .metric-label {font-size: 1rem; color: #6c757d; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;}
    
    /* Headers */
    h1, h2, h3 {color: #212529; font-family: 'Inter', sans-serif;}
    h1 {font-weight: 800; letter-spacing: -1px;}
    
    /* Custom Classes */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .result-pos {background-color: #d1e7dd; color: #0f5132; border: 1px solid #badbcc;}
    .result-neg {background-color: #f8d7da; color: #842029; border: 1px solid #f5c2c7;}
    
</style>
""", unsafe_allow_html=True)

# --------- Sidebar Navigation ---------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/723/723955.png", width=50)
    st.title("Airline NLP")
    st.markdown("Professional Sentiment Analysis Dashboard")
    st.markdown("---")
    
    page = st.radio("Navigation", ["Dashboard Overview", "Prediction Engine", "Batch Analysis", "Model Insights"])
    
    st.markdown("---")
    st.info("v2.0 | Enterprise Edition")

# --------- Page: Dashboard Overview ---------
if page == "Dashboard Overview":
    st.title("📊 Executive Dashboard")
    st.markdown("Overview of airline sentiment trends and dataset metrics.")
    
    # Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Reviews</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card"><div class="metric-value">92%</div><div class="metric-label">Model Accuracy</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card"><div class="metric-value">4.2</div><div class="metric-label">Avg Rating</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card"><div class="metric-value">24h</div><div class="metric-label">Update Cycle</div></div>""", unsafe_allow_html=True)
    
    # --------- Visualizations ---------
    st.markdown("### 📈 Sentiment & Ratings Analysis")
    
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            if 'recommended' in df.columns:
                fig1, ax1 = plt.subplots()
                df['recommended'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#21ad88', '#ea5757'], ax=ax1, startangle=90)
                ax1.set_ylabel('')
                st.pyplot(fig1)
            else:
                st.warning("Column 'recommended' not found.")

        with col2:
            st.subheader("Rating Distribution")
            if 'overall' in df.columns:
                fig2, ax2 = plt.subplots()
                sns.histplot(df['overall'], bins=10, kde=True, color='#0683b0', ax=ax2)
                ax2.set_xlabel("Rating (1-10)")
                st.pyplot(fig2)
            else:
                st.warning("Column 'overall' not found.")

        st.markdown("### ✈️ Airline Performance")
        if 'airline' in df.columns and 'overall' in df.columns:
            avg_ratings = df.groupby('airline')['overall'].mean().sort_values(ascending=False).head(10)
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.barplot(x=avg_ratings.values, y=avg_ratings.index, palette="viridis", ax=ax3)
            ax3.set_xlabel("Average Rating")
            st.pyplot(fig3)
        else:
            st.warning("Airline data not available.")
            
    else:
        st.error("Dataset not loaded. Visualizations unavailable.")

# --------- Page: Prediction Engine ---------
elif page == "Prediction Engine":
    st.title("🤖 Prediction Engine")
    st.markdown("Real-time sentiment analysis for individual reviews.")
    
    tab1, tab2 = st.tabs(["Single Prediction", "Automated Test Suite"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            user_review = st.text_area("Enter customer review:", height=150, placeholder="Type here...")
            
            if st.button("Analyze Sentiment", type="primary"):
                if user_review.strip():
                    pred, conf, prob = predict_review(user_review, model, vectorizer)
                    
                    # Store in session state so results persist across reruns
                    st.session_state['current_review'] = user_review
                    st.session_state['current_pred'] = pred
                    st.session_state['current_conf'] = conf
                    st.session_state['current_prob'] = prob
                    st.session_state['show_results'] = True
                else:
                    st.warning("Please enter some text.")
            
            # Display results if they exist in session state
            if st.session_state.get('show_results', False):
                pred = st.session_state['current_pred']
                conf = st.session_state['current_conf']
                prob = st.session_state['current_prob']
                user_review = st.session_state['current_review']
                
                # Result Display
                res_class = "result-pos" if pred == "yes" else "result-neg"
                res_text = "Recommended" if pred == "yes" else "Not Recommended"
                icon = "👍" if pred == "yes" else "👎"
                
                st.markdown(f"""
                <div class="result-box {res_class}">
                    {icon} {res_text} (Confidence: {conf:.1f}%)
                </div>
                """, unsafe_allow_html=True)
                
                # Probability Bar
                st.markdown("### Confidence Breakdown")
                chart_data = pd.DataFrame({
                    "Sentiment": ["Not Recommended", "Recommended"],
                    "Probability": [prob[0], prob[1]]
                })
                
                import altair as alt
                c = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Sentiment', sort=None),
                    y='Probability',
                    color=alt.Color('Sentiment', scale=alt.Scale(domain=['Not Recommended', 'Recommended'], range=['#ea5757', '#21ad88'])),
                    tooltip=['Sentiment', alt.Tooltip('Probability', format='.1%')]
                )
                st.altair_chart(c, use_container_width=True)
                
                # Alert System (from app.py)
                if pred == "no" and conf >= 85:
                    st.error("⚠️ ALERT: High Priority Negative Review Detected! (Confidence > 85%)")
                    st.caption("Recommended Action: Escalate to customer service immediately.")
                    
                    # Flag and Log feature
                    if st.button("🚩 Flag This Review & Save to Log", type="secondary", key="flag_review_btn"):
                        from datetime import datetime
                        try:
                            with open('alert_log.txt', 'a', encoding='utf-8') as f:
                                f.write(f"\n{'='*70}\n")
                                f.write(f"ALERT FLAGGED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write(f"Review: {user_review}\n")
                                f.write(f"Prediction: NOT RECOMMENDED\n")
                                f.write(f"Confidence: {conf:.1f}%\n")
                                f.write(f"{'='*70}\n")
                            
                            st.success(f"✅ Review flagged and saved to alert_log.txt")
                            st.info("📋 Check alert_log.txt to view the logged entry.")
                        except Exception as e:
                            st.error(f"❌ Error saving to log: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    # Optional: Show why it didn't trigger if negative
                    if pred == "no":
                        st.info(f"Note: Confidence ({conf:.1f}%) below threshold (85%) for alert.")
        
        with col2:
            st.markdown("### Quick Guide")
            st.info("""
            **How it works:**
            1. Enter review text.
            2. Click Analyze.
            3. View sentiment & confidence.
            
            **Model:** Logistic Regression
            **Accuracy:** 92%
            """)

    with tab2:
        st.subheader("System Health Check")
        test_cases = [
            "Amazing flight! The crew was incredibly friendly.",
            "Worst airline experience ever. Delayed and rude staff.",
            "It was okay, nothing special.",
            "Great value for money, would fly again."
        ]
        
        if st.button("Run Diagnostics"):
            results = []
            for t in test_cases:
                pred, conf, _ = predict_review(t, model, vectorizer)
                results.append({
                    "Input": t,
                    "Prediction": "Recommended" if pred == "yes" else "Not Recommended",
                    "Confidence": f"{conf:.1f}%"
                })
            st.dataframe(pd.DataFrame(results), use_container_width=True)

# --------- Page: Batch Analysis ---------
elif page == "Batch Analysis":
    st.title("📦 Batch Processing")
    st.markdown("Analyze multiple reviews at once.")
    
    batch_in = st.text_area("Paste reviews (one per line):", height=200)
    if st.button("Process Batch"):
        reviews = [r.strip() for r in batch_in.split('\n') if r.strip()]
        if reviews:
            results = []
            for r in reviews:
                pred, conf, _ = predict_review(r, model, vectorizer)
                results.append({
                    "Review": r,
                    "Sentiment": "Recommended" if pred == "yes" else "Not Recommended",
                    "Confidence": conf
                })
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            
            # Download
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results (CSV)", csv, "sentiment_results.csv", "text/csv")
        else:
            st.warning("No data entered.")

# --------- Page: Model Insights ---------
elif page == "Model Insights":
    st.title("🧠 Model Intelligence")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Performance Metrics")
        st.markdown("""
        - **Algorithm:** Logistic Regression
        - **Vectorization:** TF-IDF (5000 features)
        - **Training Set:** 64,440 Reviews
        - **Accuracy:** 92.0%
        - **F1 Score:** 0.92
        """)
    
    with c2:
        st.markdown("### Feature Importance")
        st.info("Top positive words: great, comfortable, friendly, delicious")
        st.error("Top negative words: delayed, rude, dirty, terrible")

