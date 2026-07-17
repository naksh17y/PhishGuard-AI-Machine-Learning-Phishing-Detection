import pandas as pd
import numpy as np
import tldextract
import math
import re
from collections import Counter
import os

# ---------------------------------------------------------
# 1. Feature Extraction Functions
# ---------------------------------------------------------

def calculate_entropy(text):
    """Calculates Shannon Entropy to detect random character strings (DGA domains)."""
    if not text:
        return 0
    counts = Counter(text)
    entropy = 0
    for count in counts.values():
        p = count / len(text)
        entropy -= p * math.log2(p)
    return entropy

def contains_ip_address(url):
    """Checks if the URL is an IP address instead of a domain name."""
    ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    return 1 if ip_pattern.search(url) else 0

def extract_features(url, label):
    """Parses a single URL and extracts all numerical/boolean features."""
    url = str(url).lower()
    
    # Extract domain components safely
    ext = tldextract.extract(url)
    domain = ext.domain
    subdomain = ext.subdomain
    full_domain = f"{subdomain}.{domain}.{ext.suffix}" if subdomain else f"{domain}.{ext.suffix}"
    
    # Suspicious keywords common in phishing
    suspicious_words = ['login', 'verify', 'update', 'secure', 'account', 'banking', 'auth', 'support']
    has_suspicious_word = 1 if any(word in url for word in suspicious_words) else 0
    
    features = {
        'url_length': len(url),
        'domain_length': len(domain),
        'subdomain_count': len(subdomain.split('.')) if subdomain else 0,
        'entropy': calculate_entropy(full_domain),
        'special_char_count': sum(url.count(c) for c in ['@', '-', '_', '=', '?']),
        'has_ip': contains_ip_address(url),
        'has_suspicious_word': has_suspicious_word,
        'label': label # 1 for Phishing, 0 for Benign
    }
    return features

# ---------------------------------------------------------
# 2. Pipeline Execution
# ---------------------------------------------------------

def build_dataset(phish_csv_path, benign_csv_path, output_path):
    print("Checking for data files...")
    if not os.path.exists(phish_csv_path):
        print(f"ERROR: Could not find {phish_csv_path}")
        return
    if not os.path.exists(benign_csv_path):
        print(f"ERROR: Could not find {benign_csv_path}")
        return

    print("Loading raw datasets...")
    # Load PhishTank data
    phish_df = pd.read_csv(phish_csv_path)
    # Handle if Kaggle CSV has a different column name for URLs
    url_col = 'url' if 'url' in phish_df.columns else phish_df.columns[0]
    print(f"Loaded {len(phish_df)} malicious URLs.")

    # Load Tranco data (Tranco CSV has no headers)
    benign_df = pd.read_csv(benign_csv_path, header=None, names=['rank', 'url'], usecols=['url'])
    print(f"Loaded {len(benign_df)} benign URLs.")

    print("\nExtracting features (this may take a minute or two)...")
    
    # Process Phishing URLs (Label = 1)
    phish_features = [extract_features(u, 1) for u in phish_df[url_col].dropna()]
    
    # Process Benign URLs (Label = 0). Sample it to match phishing size!
    sample_size = len(phish_features)
    benign_subset = benign_df['url'].dropna().sample(n=sample_size, random_state=42)
    benign_features = [extract_features(u, 0) for u in benign_subset]

    # Combine, shuffle, and save
    print("\nCombining and saving dataset...")
    final_dataset = pd.DataFrame(phish_features + benign_features)
    final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_dataset.to_csv(output_path, index=False)
    print(f"Success! Engineered dataset saved to {output_path}")
    print("\nDataset Preview:")
    print(final_dataset.head())

# ---------------------------------------------------------
# 3. TRIGGER THE SCRIPT
# ---------------------------------------------------------
if __name__ == "__main__":
    # Get the exact folder path where this script is saved
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # Ensure the data folder exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data folder at: {data_dir}")
        print("Please put online-valid.csv and top-1m.csv inside it!")
        
    build_dataset(
        phish_csv_path=os.path.join(data_dir, 'online-valid.csv'), 
        benign_csv_path=os.path.join(data_dir, 'top-1m.csv'), 
        output_path=os.path.join(data_dir, 'phishing_features.csv')
    )
    