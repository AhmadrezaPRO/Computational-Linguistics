import pandas as pd
import numpy as np
import joblib
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# --- ADD THIS LINE FOR 100% REPRODUCIBILITY ---
random.seed(42)

# --- MASTER KEYWORD LIST (Must match Validation) ---
HIGH_RISK_KEYWORDS = [
    "heart attack", "chest pain", "stroke", "paralysis", "slurred speech",
    "can't breathe", "shortness of breath", "choking", "turning blue",
    "unconscious", "seizure", "vision lost", "severe pain", "crushing",
    "bleeding heavily", "vomiting blood", "deep cut", "head injury",
    "suicide", "kill myself", "want to die", "end it all", "overdose"
]

LOW_RISK_KEYWORDS = [
    "rash", "acne", "pimple", "itch", "skin tag", "bump", "hair loss", 
    "dry skin", "sunburn", "scar",
    "runny nose", "sore throat", "mild fever", "cough", "cold", "flu", 
    "stomach ache", "constipation", "diarrhea", "headache", "migraine",
    "vitamin", "supplement", "weight", "diet", "sleep", "insomnia",
    "anxious", "worried", "sad", "depressed", "lonely", "stress",
    "question", "wondering", "curious", "pregnant", "period"
]

def generate_training_data():
    data = []
    
    # --- NOISE TEMPLATES (Mimic Real User Input) ---
    # Real users add greetings, age/gender, and filler words.
    intros = [
        "Hi doctor,", "Hello,", "I am a 25 year old male and", "My mother is 60 and",
        "Just wondering,", "I have a question:", "Please help,", "I'm scared because"
    ]
    middles = [
        "I have", "experiencing", "suffering from", "dealing with", 
        "noticed some", "worried about this"
    ]
    outros = [
        "is this bad?", "what should I do?", "please advise.", "it hurts.",
        "since yesterday.", "for 2 weeks.", "right now."
    ]

    # GENERATE HIGH RISK
    for kw in HIGH_RISK_KEYWORDS:
        for _ in range(60): # 60 examples per keyword
            text = f"{random.choice(intros)} {random.choice(middles)} {kw} {random.choice(outros)}"
            data.append((text, "High Risk"))

    # GENERATE LOW RISK
    for kw in LOW_RISK_KEYWORDS:
        for _ in range(60): 
            text = f"{random.choice(intros)} {random.choice(middles)} {kw} {random.choice(outros)}"
            data.append((text, "Low Risk"))

    return pd.DataFrame(data, columns=["text", "emotion"])

# --- TRAINING ---
print("--- TRAINING SYNCHRONIZED CLASSIFIER ---")
df = generate_training_data()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Training on {len(df)} synthetic examples matching validation vocabulary.")

pipeline = Pipeline([
    # N-grams=2 helps capture "chest pain" vs just "pain"
    ('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', CalibratedClassifierCV(LinearSVC(class_weight='balanced'), cv=5))
])

pipeline.fit(df['text'], df['emotion'])
joblib.dump(pipeline, "emotion_classifier.pkl")
print("✅ Model Saved! (Vocabulary Synchronized)")