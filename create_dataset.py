import os, re, csv, cv2
import pytesseract
from pathlib import Path

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DATASET_DIR = Path('dataset')
OUTPUT_CSV  = 'payment_features.csv'
HEADER = ['image_path','label','amount_present','txnid_present','status_positive',
    'status_present','platform_detected','date_present','text_length','word_count','digit_ratio','amount_repeat']

def ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(denoised,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    h,w = thresh.shape
    resized = cv2.resize(thresh,(w*2,h*2),interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(resized, config='--oem 3 --psm 6', lang='eng')

def features(path, label):
    img = cv2.imread(str(path))
    if img is None: return None
    text = ocr(img)
    amount = None
    for p in [r'Rs\.?\s*([\d,]+(?:\.\d{2})?)',r'NPR\.?\s*([\d,]+(?:\.\d{2})?)',r'Amount[:\s]+([\d,]+(?:\.\d{2})?)']:
        m = re.search(p, text, re.IGNORECASE)
        if m: amount = m.group(1).replace(',',''); break
    txn = re.search(r'\b([A-Z0-9]{8,20})\b', text)
    status = re.search(r'\b(Success|Successful|Completed|Pending|Failed|Processing)\b', text, re.IGNORECASE)
    date = re.search(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}', text)
    plat = re.search(r'esewa|e-sewa|khalti', text, re.IGNORECASE)
    s = status.group(0).lower() if status else None
    pos = {'success','successful','completed','complete'}
    return {'image_path':str(path),'label':label,
        'amount_present':1 if amount else 0,'txnid_present':1 if txn else 0,
        'status_positive':1 if s in pos else 0,'status_present':1 if s else 0,
        'platform_detected':1 if plat else 0,'date_present':1 if date else 0,
        'text_length':min(len(text),1000)/1000,'word_count':min(len(text.split()),200)/200,
        'digit_ratio':sum(c.isdigit() for c in text)/max(len(text),1),
        'amount_repeat':text.count(amount) if amount else 0}

def process(folder, label, rows):
    exts = {'.png','.jpg','.jpeg','.webp'}
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    print(f"  {len(imgs)} images in '{folder.name}'")
    for p in imgs:
        print(f"    {p.name}...", end=' ', flush=True)
        f = features(p, label)
        if f: rows.append(f); print("OK")
        else: print("skip")

def main():
    genuine_dir, fake_dir = DATASET_DIR/'genuine', DATASET_DIR/'fake'
    if not genuine_dir.exists() or not fake_dir.exists():
        print("\n[!] Create folders: dataset/genuine/ and dataset/fake/"); return
    rows = []
    print("\n[*] Processing genuine screenshots...")
    process(genuine_dir, 0, rows)
    print("\n[*] Processing fake screenshots...")
    process(fake_dir, 1, rows)
    if not rows: print("\n[!] No images found."); return
    with open(OUTPUT_CSV,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader(); w.writerows(rows)
    print(f"\n[OK] Saved {len(rows)} samples to {OUTPUT_CSV}")
    print(f"     Genuine: {sum(1 for r in rows if r['label']==0)} | Fake: {sum(1 for r in rows if r['label']==1)}")
    print("\nNext: python train_model.py\n")

if __name__ == '__main__':
    main()
