import pandas as pd
from datasets import load_dataset

# --- CONFIGURATION ---
HF_DATASET = "ruslanmv/ai-medical-chatbot"

# --- TRIGGERS ---
HIGH_RISK_TRIGGERS = ["heart attack", "chest pain", "stroke", "paralysis", "slurred speech", "can't breathe", "shortness of breath", "choking", "turning blue", "unconscious", "seizure", "vision lost", "severe pain", "crushing", "bleeding heavily", "vomiting blood", "deep cut", "head injury", "suicide", "kill myself", "want to die", "end it all", "overdose"]
LOW_RISK_TRIGGERS = ["rash", "acne", "pimple", "itch", "skin tag", "bump", "hair loss", "dry skin", "sunburn", "scar", "runny nose", "sore throat", "mild fever", "cough", "cold", "flu", "stomach ache", "constipation", "diarrhea", "headache", "migraine", "vitamin", "supplement", "weight", "diet", "sleep", "insomnia", "anxious", "worried", "sad", "depressed", "lonely", "stress", "question", "wondering", "curious", "pregnant", "period"]

def generate_dataset(total_count=100):
    half = total_count // 2
    print(f"⬇️ Downloading data and searching for {total_count} scenarios...")
    try:
        dataset = load_dataset(HF_DATASET, split="train", streaming=True)
        results = []
        for i, row in enumerate(dataset):
            if i > 50000: break # Scan deeper to ensure 100 unique cases
            text = row.get('Patient', '')
            summary = row.get('Description', '') or (text[:50] + "...")
            text_lower = text.lower()
            
            label = None
            if any(k in text_lower for k in HIGH_RISK_TRIGGERS): label = "High Risk"
            elif any(k in text_lower for k in LOW_RISK_TRIGGERS): label = "Low Risk"
            
            if label:
                results.append({"text": text, "summary": summary, "true_label": label})

        df = pd.DataFrame(results).drop_duplicates(subset=['text'])
        
        # Pull 50/50 split
        df_low = df[df['true_label'] == "Low Risk"].sample(n=half, random_state=42)
        df_high = df[df['true_label'] == "High Risk"].sample(n=half, random_state=42)
        
        # Combine: Low (1-50), High (51-100)
        final_df = pd.concat([df_low, df_high]).reset_index(drop=True)
        final_df.to_csv("validation_dataset.csv", index=False)
        print(f"✅ Success! 1-50 are Low Risk, 51-100 are High Risk.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    generate_dataset(100)