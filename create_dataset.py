import os
import cv2
import pandas as pd
import pytesseract
import re
from PIL import Image
import numpy as np

# Set Tesseract path (Windows)
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(img_path):
    """Read and preprocess image for OCR"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    h, w = thresh.shape
    return cv2.resize(thresh, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

def extract_features(img_path):
    """Extract features from image for ML model"""
    img = preprocess_image(img_path)
    if img is None:
        return None
    
    # OCR text
    text = pytesseract.image_to_string(img, config='--oem 3 --psm 6', lang='eng')
    
    # Extract payment fields
    amount = None
    patterns = [r'Rs\.?\s*([\d,]+(?:\.\d{2})?)', r'NPR\.?\s*([\d,]+(?:\.\d{2})?)']
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            amount = m.group(1).replace(',', '')
            break
    
    txn = re.search(r'\b([A-Z0-9]{8,20})\b', text)
    status = re.search(r'\b(Success|Successful|Completed|Pending|Failed)\b', text, re.IGNORECASE)
    platform = None
    if re.search(r'esewa|e-sewa', text, re.IGNORECASE):
        platform = 'eSewa'
    elif re.search(r'khalti', text, re.IGNORECASE):
        platform = 'Khalti'
    
    # Build feature vector
    features = [
        1 if amount else 0,
        1 if txn else 0,
        1 if status and status.group(0).lower() in ['success', 'successful', 'completed'] else 0,
        1 if status else 0,
        1 if platform else 0,
        1 if re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', text) else 0,
        min(len(text), 1000) / 1000,
        min(len(text.split()), 200) / 200,
        sum(c.isdigit() for c in text) / max(len(text), 1),
        text.count(amount) if amount else 0
    ]
    
    return features

def create_dataset():
    """Process all images and create dataset CSV"""
    data = []
    labels = []
    
    # Process genuine images
    genuine_path = 'dataset/genuine/'
    if os.path.exists(genuine_path):
        for img_file in os.listdir(genuine_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(genuine_path, img_file)
                features = extract_features(img_path)
                if features:
                    data.append(features)
                    labels.append(0)  # 0 = genuine
    
    # Process fake images
    fake_path = 'dataset/fake/'
    if os.path.exists(fake_path):
        for img_file in os.listdir(fake_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(fake_path, img_file)
                features = extract_features(img_path)
                if features:
                    data.append(features)
                    labels.append(1)  # 1 = fake
    
    # Save to CSV
    if data and labels:
        df = pd.DataFrame(data, columns=[
            'has_amount', 'has_txn', 'has_success_status', 'has_status',
            'has_platform', 'has_date', 'text_length', 'word_count',
            'digit_ratio', 'amount_frequency'
        ])
        df['label'] = labels
        df.to_csv('dataset_features.csv', index=False)
        print(f"✅ Dataset created! {len(data)} images processed.")
        print(f"   Genuine: {len([l for l in labels if l == 0])}")
        print(f"   Fake: {len([l for l in labels if l == 1])}")
    else:
        print("❌ No images found! Add images to dataset/genuine/ and dataset/fake/")
        print("   Genuine path:", genuine_path)
        print("   Fake path:", fake_path)

if __name__ == '__main__':
    create_dataset()