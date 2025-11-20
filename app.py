"""
Airline Review Sentiment Classifier 
Terminal-based NLP demonstration system
"""

import pickle
import string
import os
from datetime import datetime
import time


def load_model():
    """Load trained model"""
    print("\n" + "="*70)
    print("   AIRLINE REVIEW SENTIMENT ANALYSIS SYSTEM")
    print("="*70)
    print("\nInitializing system...")
    
    for i in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    
    model = pickle.load(open('airline_review_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    
    print("\n\nModel loaded: LogisticRegression")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())} words")
    print(f"Model accuracy: 92%")
    print(f"System ready at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return model, vectorizer


model, vectorizer = load_model()


def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation + '0123456789'))
    return ' '.join(text.split())


def predict_review(review_text):
    """Predict sentiment with confidence score"""
    cleaned = clean_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(vectorized)[0]
        confidence = max(prob) * 100
        return prediction, confidence, prob
    return prediction, None, None


def display_result(prediction, confidence, prob, review_text):
    """Display detailed prediction results"""
    print("\n" + "="*70)
    print("   PREDICTION RESULTS")
    print("="*70)
    
    print(f"\nReview: \"{review_text[:100]}{'...' if len(review_text) > 100 else ''}\"")
    print(f"Length: {len(review_text.split())} words")
    
    print("\n" + "-"*70)
    
    if prediction == "yes":
        print("   SENTIMENT: RECOMMENDED")
        print("   Customer is satisfied with the airline")
    else:
        print("   SENTIMENT: NOT RECOMMENDED")
        print("   Customer is dissatisfied with the airline")
    
    if confidence:
        print(f"\n   Confidence Score: {confidence:.1f}%")
        
        # Confidence bar
        bar_length = int(confidence / 2)
        bar = "=" * bar_length + "-" * (50 - bar_length)
        print(f"   [{bar}] {confidence:.1f}%")
        
        # Interpretation
        if confidence >= 90:
            print("   Very High Confidence - Strong signal")
        elif confidence >= 80:
            print("   High Confidence - Clear signal")
        elif confidence >= 70:
            print("   Good Confidence - Reliable prediction")
        elif confidence >= 60:
            print("   Moderate Confidence - Acceptable")
        else:
            print("   Low Confidence - Ambiguous signal")
        
        # Probability breakdown
        if prob is not None:
            print(f"\n   Probability Breakdown:")
            print(f"      Recommended: {prob[1]*100:.1f}%")
            print(f"      Not Recommended: {prob[0]*100:.1f}%")
    
    print("="*70 + "\n")


def run_test_examples():
    """Run comprehensive test suite"""
    print("\n" + "="*70)
    print("   RUNNING TEST SUITE")
    print("="*70)
    
    test_cases = [
        {
            'category': 'Positive - Excellent Service',
            'review': 'Amazing flight! The crew was incredibly friendly and helpful. '
                     'Comfortable seats and delicious food. Best airline experience I have ever had. Highly recommend!'
        },
        {
            'category': 'Negative - Poor Service',
            'review': 'Worst airline experience ever. Flight delayed by 5 hours with absolutely no explanation. '
                     'Staff were rude and unhelpful. Uncomfortable seats and terrible food. Never flying with them again!'
        },
        {
            'category': 'Neutral - Mixed Experience',
            'review': 'Decent flight overall. Nothing exceptional but got me to my destination safely. '
                     'Service was average, seats were okay. Price was reasonable for what you get.'
        },
        {
            'category': 'Positive - Great Value',
            'review': 'Excellent value for money. Clean plane, smooth flight, and professional staff. '
                     'In-flight entertainment was great. Would definitely fly with them again!'
        },
        {
            'category': 'Negative - Multiple Issues',
            'review': 'Terrible experience from start to finish. Lost my baggage, poor customer service response, '
                     'plane was dirty and cramped. Food was inedible. Avoid this airline at all costs!'
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
    
    results_summary = {'recommended': 0, 'not_recommended': 0}
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test['category']} ---")
        print(f"Review: \"{test['review'][:80]}...\"" if len(test['review']) > 80 else f"Review: \"{test['review']}\"")
        
        prediction, confidence, prob = predict_review(test['review'])
        
        if prediction == "yes":
            result_text = "RECOMMENDED"
            results_summary['recommended'] += 1
        else:
            result_text = "NOT RECOMMENDED"
            results_summary['not_recommended'] += 1
        
        print(f"Result: {result_text} (Confidence: {confidence:.1f}%)")
        print("-" * 70)
    
    # Summary
    print("\n" + "="*70)
    print("   TEST SUITE SUMMARY")
    print("="*70)
    print(f"   Total Tests: {len(test_cases)}")
    print(f"   Recommended: {results_summary['recommended']}")
    print(f"   Not Recommended: {results_summary['not_recommended']}")
    print("="*70)


def batch_analysis():
    """Analyze multiple reviews at once"""
    print("\n" + "="*70)
    print("   BATCH ANALYSIS MODE")
    print("="*70)
    print("\nEnter multiple reviews (one per line).")
    print("When done, type 'DONE' on a new line.\n")
    
    reviews = []
    while True:
        review = input(f"Review {len(reviews) + 1}: ")
        if review.upper() == 'DONE':
            break
        if review.strip():
            reviews.append(review)
    
    if not reviews:
        print("No reviews entered.")
        return
    
    print(f"\nAnalyzing {len(reviews)} reviews...\n")
    
    results = []
    for review in reviews:
        pred, conf, prob = predict_review(review)
        results.append({
            'review': review,
            'prediction': pred,
            'confidence': conf,
            'is_recommended': pred == 'yes'
        })
    
    # Statistics
    recommended = sum(1 for r in results if r['is_recommended'])
    not_recommended = len(results) - recommended
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    # Display summary
    print("="*70)
    print("   BATCH ANALYSIS RESULTS")
    print("="*70)
    print(f"\nSummary Statistics:")
    print(f"   Total Reviews: {len(results)}")
    print(f"   Recommended: {recommended} ({recommended/len(results)*100:.1f}%)")
    print(f"   Not Recommended: {not_recommended} ({not_recommended/len(results)*100:.1f}%)")
    print(f"   Average Confidence: {avg_confidence:.1f}%")
    
    # Detailed results
    print(f"\nDetailed Results:")
    print("-"*70)
    for i, result in enumerate(results, 1):
        status = "REC" if result['is_recommended'] else "NOT"
        review_short = result['review'][:50] + "..." if len(result['review']) > 50 else result['review']
        print(f"{i}. {status} ({result['confidence']:.1f}%) | {review_short}")
    print("="*70)


def airline_comparison():
    """Compare sentiment across different airlines"""
    print("\n" + "="*70)
    print("   AIRLINE COMPARISON DASHBOARD")
    print("="*70)
    
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
    
    print("\nAnalyzing customer sentiment across airlines...\n")
    
    comparison_results = {}
    
    for airline, reviews in airlines_data.items():
        predictions = []
        confidences = []
        
        for review in reviews:
            pred, conf, _ = predict_review(review)
            predictions.append(pred == 'yes')
            confidences.append(conf)
        
        recommended_pct = (sum(predictions) / len(predictions)) * 100
        avg_conf = sum(confidences) / len(confidences)
        
        comparison_results[airline] = {
            'recommended_pct': recommended_pct,
            'avg_confidence': avg_conf,
            'total_reviews': len(reviews)
        }
    
    # Display results
    print("="*70)
    for airline, stats in comparison_results.items():
        print(f"\n{airline}")
        print(f"   Positive Sentiment: {stats['recommended_pct']:.0f}%")
        print(f"   Average Confidence: {stats['avg_confidence']:.1f}%")
        print(f"   Sample Size: {stats['total_reviews']} reviews")
        
        # Rating
        if stats['recommended_pct'] >= 80:
            rating = "Excellent (5 stars)"
        elif stats['recommended_pct'] >= 60:
            rating = "Very Good (4 stars)"
        elif stats['recommended_pct'] >= 40:
            rating = "Average (3 stars)"
        else:
            rating = "Below Average (2 stars)"
        
        print(f"   Overall Rating: {rating}")
        print("-"*70)
    
    print("="*70)


def interactive_mode():
    """Interactive mode with history tracking"""
    print("\n" + "="*70)
    print("   INTERACTIVE PREDICTION MODE")
    print("="*70)
    print("\nCommands:")
    print("  - Type a review to analyze")
    print("  - 'stats' to see session statistics")
    print("  - 'clear' to clear history")
    print("  - 'quit' or 'exit' to return to main menu")
    print("="*70 + "\n")
    
    history = []
    
    while True:
        # Show recent history
        if history:
            print("\n" + "-"*70)
            print("   RECENT PREDICTIONS (Last 5)")
            print("-"*70)
            recent = history[-5:]
            for i, item in enumerate(recent, len(history)-len(recent)+1):
                status = "REC" if item['pred'] == 'yes' else "NOT"
                review_short = item['review'][:45] + "..." if len(item['review']) > 45 else item['review']
                print(f"  {i}. {status} ({item['conf']:.1f}%) | {review_short}")
            if len(history) > 5:
                print(f"  ... and {len(history)-5} more (type 'stats' for full history)")
            print("-"*70 + "\n")
        
        review = input("Enter review (or command): ")
        
        # Handle commands
        if review.lower() in ['quit', 'exit', 'q']:
            if history:
                rec_count = sum(1 for h in history if h['pred'] == 'yes')
                print("\n" + "="*70)
                print("   SESSION SUMMARY")
                print("="*70)
                print(f"   Total Predictions: {len(history)}")
                print(f"   Recommended: {rec_count} ({rec_count/len(history)*100:.1f}%)")
                print(f"   Not Recommended: {len(history)-rec_count}")
                print("="*70)
            print("\nReturning to main menu...\n")
            break
        
        if review.lower() == 'clear':
            history.clear()
            print("History cleared.\n")
            continue
        
        if review.lower() == 'stats':
            if not history:
                print("No predictions yet.\n")
            else:
                rec_count = sum(1 for h in history if h['pred'] == 'yes')
                avg_conf = sum(h['conf'] for h in history) / len(history)
                
                print("\n" + "="*70)
                print("   DETAILED SESSION STATISTICS")
                print("="*70)
                print(f"\n   Total Predictions: {len(history)}")
                print(f"   Recommended: {rec_count} ({rec_count/len(history)*100:.1f}%)")
                print(f"   Not Recommended: {len(history)-rec_count} ({(len(history)-rec_count)/len(history)*100:.1f}%)")
                print(f"   Average Confidence: {avg_conf:.1f}%")
                
                print(f"\n   Complete History:")
                print("   " + "-"*66)
                for i, item in enumerate(history, 1):
                    status = "REC" if item['pred'] == 'yes' else "NOT"
                    review_short = item['review'][:40] + "..." if len(item['review']) > 40 else item['review']
                    print(f"   {i:2d}. {status} ({item['conf']:4.1f}%) | {review_short}")
                print("="*70 + "\n")
            continue
        
        # Validate input
        if not review.strip():
            print("Please enter a valid review.\n")
            continue
        
        if len(review.split()) < 2:
            print("Review too short. Please provide more detail.\n")
            continue
        
        # Make prediction
        prediction, confidence, prob = predict_review(review)
        
        # Display result
        print("\n" + "="*70)
        print("   PREDICTION RESULT")
        print("="*70)
        
        print(f"\nReview: \"{review[:60]}{'...' if len(review) > 60 else ''}\"")
        
        print("\n" + "-"*70)
        
        if prediction == "yes":
            print("   SENTIMENT: RECOMMENDED")
            print("   Customer is satisfied")
        else:
            print("   SENTIMENT: NOT RECOMMENDED")
            print("   Customer is dissatisfied")
        
        if confidence:
            print(f"\n   Confidence Score: {confidence:.1f}%")
            
            # Confidence bar
            bar_length = int(confidence / 2)
            bar = "=" * bar_length + "-" * (50 - bar_length)
            print(f"   [{bar}] {confidence:.1f}%")
            
            # Interpretation
            if confidence >= 90:
                print("   Very High Confidence")
            elif confidence >= 80:
                print("   High Confidence")
            elif confidence >= 70:
                print("   Good Confidence")
            elif confidence >= 60:
                print("   Moderate Confidence")
            else:
                print("   Low Confidence")
        
        print("="*70)
        
        # Alert system
        if prediction == "no" and confidence and confidence >= 85:
            print("\n" + "!"*70)
            print("   ALERT: HIGH PRIORITY NEGATIVE REVIEW DETECTED")
            print("!"*70)
            print("\n   Alert Details:")
            print(f"   - Sentiment: Negative")
            print(f"   - Confidence: {confidence:.1f}% (Very High)")
            print(f"   - Priority Level: URGENT")
            print(f"   - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\n   Recommended Actions:")
            print("   1. Notify customer service manager")
            print("   2. Contact customer within 24 hours")
            print("   3. Investigate issues mentioned in review")
            print("   4. Document in CRM system")
            print("\n" + "!"*70 + "\n")
            
            # Log alert
            with open('alert_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"ALERT TRIGGERED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Review: {review}\n")
                f.write(f"Prediction: NOT RECOMMENDED\n")
                f.write(f"Confidence: {confidence:.1f}%\n")
                f.write(f"{'='*70}\n")
        
        # Add to history
        history.append({
            'review': review,
            'pred': prediction,
            'conf': confidence
        })
        
        # Quick stats
        rec_count = sum(1 for h in history if h['pred'] == 'yes')
        print(f"\nSession: {len(history)} predictions | {rec_count} recommended | Avg: {sum(h['conf'] for h in history)/len(history):.1f}%")


def main_menu():
    """Main application menu"""
    while True:
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("\n1. Run Test Suite (7 test cases)")
        print("2. Interactive Prediction Mode")
        print("3. Batch Analysis")
        print("4. Airline Comparison Dashboard")
        print("5. View Model Information")
        print("6. Exit System")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            run_test_examples()
            input("\nPress Enter to continue...")
        elif choice == "2":
            interactive_mode()
        elif choice == "3":
            batch_analysis()
            input("\nPress Enter to continue...")
        elif choice == "4":
            airline_comparison()
            input("\nPress Enter to continue...")
        elif choice == "5":
            print("\n" + "="*70)
            print("MODEL INFORMATION")
            print("="*70)
            print(f"   Model Type: Logistic Regression")
            print(f"   Feature Extraction: TF-IDF Vectorization")
            print(f"   Vocabulary Size: {len(vectorizer.get_feature_names_out())} words")
            print(f"   Training Accuracy: 92%")
            print(f"   Precision: 92%")
            print(f"   Recall: 92%")
            print(f"   F1-Score: 0.92")
            print(f"   Dataset: 64,440 airline reviews")
            print("="*70)
            input("\nPress Enter to continue...")
        elif choice == "6":
            print("\n" + "="*70)
            print("Thank you for using the Airline Review Classifier!")
            print("System shutting down...")
            print("="*70 + "\n")
            break
        else:
            print("Invalid choice. Please select 1-6.")


if __name__ == "__main__":
    main_menu()
