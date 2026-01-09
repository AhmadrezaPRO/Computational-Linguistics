import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# --- CONFIGURATION ---
MODEL_PATH = "emotion_classifier.pkl"
DATA_PATH = "validation_dataset.csv"

def test_model_performance():
    # 1. Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found. Run train_model.py first.")
        return
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file '{DATA_PATH}' not found. Run create_balanced_dataset.py first.")
        return

    print("--- 🔍 MODEL DIAGNOSTICS ---")
    
    # 2. Load Resources
    print(f"Loading model: {MODEL_PATH}...")
    classifier = joblib.load(MODEL_PATH)
    
    print(f"Loading data:  {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 3. Run Predictions
    print("\nRunning predictions on validation set...")
    inputs = df['text']
    true_labels = df['true_label']
    
    # Predict
    predictions = classifier.predict(inputs)
    
    # 4. Calculate Metrics
    acc = accuracy_score(true_labels, predictions)
    
    print("\n" + "="*30)
    print(f"🎯 OVERALL ACCURACY: {acc*100:.2f}%")
    print("="*30)
    
    # 5. Detailed Report
    print("\n📊 Classification Report:")
    print(classification_report(true_labels, predictions))
    
    # 6. Confusion Matrix (Visual check)
    cm = confusion_matrix(true_labels, predictions, labels=["High Risk", "Low Risk"])
    print("\nConfusion Matrix:")
    print(f"True High | {cm[0][0]} (Correct)  | {cm[0][1]} (Missed)")
    print(f"True Low  | {cm[1][0]} (False Alarm)| {cm[1][1]} (Correct)")

    # 7. Show Errors (Optional but helpful)
    print("\n⚠️ MISCLASSIFIED EXAMPLES:")
    df['predicted'] = predictions
    errors = df[df['true_label'] != df['predicted']]
    
    if errors.empty:
        print("None! Perfect score.")
    else:
        for i, row in errors.head(5).iterrows():
            print(f"- Text: \"{row['text'][:60]}...\"")
            print(f"  True: {row['true_label']} | Predicted: {row['predicted']}\n")

if __name__ == "__main__":
    test_model_performance()