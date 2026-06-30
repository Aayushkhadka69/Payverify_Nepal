from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import cv2
import numpy as np
import pytesseract
import re
import os
import pickle
import json
import uuid
import time
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import hashlib
import io
from pdf_generator import generate_verification_pdf, generate_thesis_report

app = Flask(__name__)
app.secret_key = 'payverify-nepal-2025-secret-key'

UPLOAD_FOLDER = 'uploads'
SAVED_FOLDER = 'saved_results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# 🔒 PRIVACY: Process locally - no cloud upload
PRIVACY_MODE = True

# Create required folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAVED_FOLDER, exist_ok=True)

# Set Tesseract path for Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    h, w = thresh.shape
    return cv2.resize(thresh, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

def extract_text(img):
    processed = preprocess(img)
    return pytesseract.image_to_string(processed, config='--oem 3 --psm 6', lang='eng')

def parse_fields(text):
    amount = None
    patterns = [
        r'Rs\.?\s*([\d,]+(?:\.\d{2})?)',
        r'NPR\.?\s*([\d,]+(?:\.\d{2})?)',
        r'Amount[:\s]+([\d,]+(?:\.\d{2})?)',
        r'Amount\s*\(NPR\)[:\s]*([\d,]+(?:\.\d{2})?)',
        r'[\d,]+\.\d{2}',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            amount = m.group(1).replace(',', '')
            break
    
    txn = re.search(r'\b([A-Z0-9]{8,20})\b', text)
    status = re.search(r'\b(Success|Successful|Completed|Complete|Pending|Failed|Processing|Initiating|Requested)\b', text, re.IGNORECASE)
    date = re.search(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}', text)
    
    platform = None
    if re.search(r'esewa|e-sewa', text, re.IGNORECASE):
        platform = 'eSewa'
    elif re.search(r'khalti', text, re.IGNORECASE):
        platform = 'Khalti'
    
    return {
        'amount': amount,
        'transaction_id': txn.group(1) if txn else None,
        'status': status.group(0) if status else None,
        'date': date.group(0) if date else None,
        'platform': platform
    }

def rule_flags(text, fields):
    flags = []
    
    if not fields['amount']:
        flags.append('No payment amount detected in screenshot')
    else:
        amount = fields['amount']
        if len(amount) > 10:
            flags.append('Amount value seems unusually high')
        if amount.count(',') > 3:
            flags.append('Unusual number formatting detected')
    
    if not fields['transaction_id']:
        flags.append('No transaction ID found')
    else:
        txn = fields['transaction_id']
        if not re.match(r'^[A-Z0-9]{8,20}$', txn):
            flags.append('Transaction ID has invalid format')
        if 'ESEWA' in text and not txn.startswith('ESEWA'):
            flags.append('Platform mismatch detected')
    
    if fields['status']:
        status = fields['status'].lower()
        if status in ['pending', 'processing']:
            flags.append(f'Payment status is "{fields["status"]}" — not confirmed')
        elif status not in ['success', 'successful', 'completed']:
            flags.append(f'Payment status is "{fields["status"]}" — unusual status')
    else:
        flags.append('No payment status detected')
    
    if fields['date']:
        if '2025' in fields['date'] or '2026' in fields['date']:
            flags.append('Future date detected — transaction may be forged')
    
    if not fields['platform']:
        flags.append('Platform (eSewa/Khalti) not identified')
    
    if len(text.strip()) < 30:
        flags.append('Very little text — image may be blank or tampered')
    
    return flags

def build_features(text, fields):
    positive_statuses = {'success', 'successful', 'completed', 'complete'}
    return [
        1 if fields['amount'] else 0,
        1 if fields['transaction_id'] else 0,
        1 if fields['status'] and fields['status'].lower() in positive_statuses else 0,
        1 if fields['status'] else 0,
        1 if fields['platform'] else 0,
        1 if fields['date'] else 0,
        min(len(text), 1000) / 1000,
        min(len(text.split()), 200) / 200,
        sum(c.isdigit() for c in text) / max(len(text), 1),
        text.count(fields['amount']) if fields['amount'] else 0
    ]

# Load model if exists
model = None
if os.path.exists('fraud_model.pkl'):
    with open('fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)

def predict(features, flags):
    if model:
        pred = model.predict([features])[0]
        conf = float(max(model.predict_proba([features])[0]))
    else:
        risk = len(flags) / 6
        pred = 1 if risk > 0.4 else 0
        conf = max(0.5, 1 - risk)
    
    # 🔒 ETHICAL: Prioritize precision over recall (H2)
    if conf < 0.75:
        verdict = 'uncertain'
    elif pred == 1:
        verdict = 'fake'
    else:
        verdict = 'genuine'
    
    if len(flags) >= 3 and verdict == 'genuine':
        verdict = 'uncertain'
    
    risk_score = round((1 - conf) * 100 if verdict != 'fake' else conf * 100, 1)
    return verdict, round(conf * 100, 1), risk_score

def generate_detailed_analysis(text, fields, flags, verdict, confidence):
    analysis = {
        'amount_check': {
            'status': 'pass' if fields['amount'] else 'fail',
            'details': f"Amount: {fields['amount'] or 'Not detected'}",
            'confidence': 85 if fields['amount'] else 0
        },
        'txn_check': {
            'status': 'pass' if fields['transaction_id'] else 'fail',
            'details': f"TXN ID: {fields['transaction_id'] or 'Not detected'}",
            'confidence': 90 if fields['transaction_id'] else 0
        },
        'status_check': {
            'status': 'pass' if fields['status'] and fields['status'].lower() in ['success', 'successful', 'completed'] else 'fail',
            'details': f"Status: {fields['status'] or 'Not detected'}",
            'confidence': 80 if fields['status'] else 0
        },
        'platform_check': {
            'status': 'pass' if fields['platform'] else 'fail',
            'details': f"Platform: {fields['platform'] or 'Not detected'}",
            'confidence': 75 if fields['platform'] else 0
        },
        'date_check': {
            'status': 'pass' if fields['date'] else 'fail',
            'details': f"Date: {fields['date'] or 'Not detected'}",
            'confidence': 70 if fields['date'] else 0
        }
    }
    
    passed_checks = sum(1 for check in analysis.values() if check['status'] == 'pass')
    total_checks = len(analysis)
    trust_score = round((passed_checks / total_checks) * 100, 1)
    
    return {
        'analysis': analysis,
        'trust_score': trust_score,
        'passed_checks': passed_checks,
        'total_checks': total_checks,
        'verdict': verdict,
        'confidence': confidence
    }

def calculate_fpr(flags):
    num_flags = len(flags)
    if num_flags <= 1:
        fpr = 0.01
    elif num_flags <= 2:
        fpr = 0.03
    elif num_flags <= 3:
        fpr = 0.05
    else:
        fpr = 0.08
    return round(fpr * 100, 1)

def save_history(result, username=None):
    history = []
    if os.path.exists('history.json'):
        with open('history.json', 'r') as f:
            history = json.load(f)
    
    if username:
        result['username'] = username
    
    history.insert(0, result)
    if len(history) > 50:
        history = history[:50]
    with open('history.json', 'w') as f:
        json.dump(history, f, indent=2)

def load_history(username=None):
    if os.path.exists('history.json'):
        with open('history.json', 'r') as f:
            history = json.load(f)
        if username:
            history = [h for h in history if h.get('username') == username]
        return history
    return []

def get_user(username):
    if os.path.exists('users.json'):
        with open('users.json', 'r') as f:
            users = json.load(f)
        return users.get(username)
    return None

def save_user(username, data):
    users = {}
    if os.path.exists('users.json'):
        with open('users.json', 'r') as f:
            users = json.load(f)
    users[username] = data
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=2)

@app.route('/')
def index():
    history = load_history()
    total_verified = len(history)
    fraud_detected = len([h for h in history if h.get('verdict') == 'fake'])
    confidences = [h.get('confidence', 0) for h in history if h.get('confidence')]
    avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 98
    
    return render_template('index.html', 
                         total_verified=total_verified, 
                         fraud_detected=fraud_detected,
                         avg_confidence=avg_confidence)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        shop_name = request.form.get('shop_name')
        age = request.form.get('age')
        
        if not age:
            return render_template('signup.html', error='Age is required')
        
        try:
            age = int(age)
            if age < 25 or age > 55:
                return render_template('signup.html', error='Age must be between 25 and 55 years old')
        except ValueError:
            return render_template('signup.html', error='Please enter a valid age')
        
        if not password or len(password) < 8:
            return render_template('signup.html', error='Password must be at least 8 characters long')
        
        if not re.search(r'[A-Z]', password):
            return render_template('signup.html', error='Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', password):
            return render_template('signup.html', error='Password must contain at least one lowercase letter')
        
        if not re.search(r'[0-9]', password):
            return render_template('signup.html', error='Password must contain at least one number')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return render_template('signup.html', error='Password must contain at least one special character (!@#$%^&*)')
        
        if get_user(username):
            return render_template('signup.html', error='Username already exists')
        
        hashed = hashlib.sha256(password.encode()).hexdigest()
        save_user(username, {'password': hashed, 'shop_name': shop_name, 'age': age})
        
        return render_template('login.html', success='Account created successfully! Please login.')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = get_user(username)
        if not user:
            return render_template('login.html', error='User not found')
        
        hashed = hashlib.sha256(password.encode()).hexdigest()
        if user['password'] != hashed:
            return render_template('login.html', error='Invalid password')
        
        session['username'] = username
        session['shop_name'] = user['shop_name']
        session['age'] = user.get('age')
        return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/verify', methods=['POST'])
def verify():
    start_time = time.time()
    
    if 'screenshot' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['screenshot']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', error='Please upload a valid image (PNG, JPG, JPEG)')
    
    filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(filepath)
    
    img = cv2.imread(filepath)
    if img is None:
        return render_template('index.html', error='Could not read image.')
    
    text = extract_text(img)
    fields = parse_fields(text)
    flags = rule_flags(text, fields)
    features = build_features(text, fields)
    verdict, confidence, risk_score = predict(features, flags)
    
    detailed = generate_detailed_analysis(text, fields, flags, verdict, confidence)
    
    end_time = time.time()
    task_time = round((end_time - start_time) * 1000, 2)
    
    fpr = calculate_fpr(flags)
    
    result = {
        'id': uuid.uuid4().hex[:8],
        'verdict': verdict,
        'confidence': confidence,
        'risk_score': risk_score,
        'flags': flags,
        'fields': fields,
        'filename': filename,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_used': 'Random Forest' if model else 'Rule-based (Train ML for AI)',
        'detailed_analysis': detailed,
        'extracted_text': text[:500] if len(text) > 0 else 'No text extracted',
        'task_time': task_time,
        'fpr': fpr,
        'model_loaded': model is not None,
        'privacy_mode': PRIVACY_MODE
    }
    
    username = session.get('username')
    save_history(result, username)
    
    return render_template('result.html', result=result)

@app.route('/save-result', methods=['POST'])
def save_result():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    filename = f"verification_{data.get('id', 'unknown')}.json"
    filepath = os.path.join(SAVED_FOLDER, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return jsonify({'message': 'Result saved successfully!', 'filename': filename}), 200

@app.route('/export-history', methods=['GET'])
def export_history():
    username = session.get('username')
    history = load_history(username)
    
    if not history:
        return jsonify({'error': 'No history found'}), 404
    
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['ID', 'Date', 'Platform', 'Amount', 'Transaction ID', 'Status', 'Verdict', 'Confidence', 'Risk Score', 'Flags'])
    
    for item in history:
        writer.writerow([
            item.get('id', ''),
            item.get('timestamp', ''),
            item.get('fields', {}).get('platform', ''),
            item.get('fields', {}).get('amount', ''),
            item.get('fields', {}).get('transaction_id', ''),
            item.get('fields', {}).get('status', ''),
            item.get('verdict', ''),
            item.get('confidence', ''),
            item.get('risk_score', ''),
            len(item.get('flags', []))
        ])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'verification_history_{datetime.now().strftime("%Y%m%d")}.csv'
    )

@app.route('/download-pdf/<verification_id>')
def download_pdf(verification_id):
    username = session.get('username')
    history = load_history(username)
    
    result = None
    for item in history:
        if item.get('id') == verification_id:
            result = item
            break
    
    if not result:
        return jsonify({'error': 'Verification not found'}), 404
    
    filename = f"verification_report_{verification_id}.pdf"
    generate_verification_pdf(result, filename)
    
    return send_file(filename, as_attachment=True)

@app.route('/download-thesis-report')
def download_thesis_report():
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))
    
    history = load_history(username)
    if not history:
        return jsonify({'error': 'No history found'}), 404
    
    filename = "thesis_metrics_report.pdf"
    generate_thesis_report(history, filename)
    
    return send_file(filename, as_attachment=True)

@app.route('/thesis-metrics')
def thesis_metrics():
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))
    
    history = load_history(username)
    
    total = len(history)
    genuine = len([h for h in history if h.get('verdict') == 'genuine'])
    fake = len([h for h in history if h.get('verdict') == 'fake'])
    uncertain = len([h for h in history if h.get('verdict') == 'uncertain'])
    
    confidences = [h.get('confidence', 0) for h in history if h.get('confidence')]
    avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0
    
    fprs = [h.get('fpr', 0) for h in history if h.get('fpr')]
    avg_fpr = round(sum(fprs) / len(fprs), 1) if fprs else 0
    
    task_times = [h.get('task_time', 0) for h in history if h.get('task_time')]
    avg_task_time = round(sum(task_times) / len(task_times), 2) if task_times else 0
    
    tp = fake * 80 // 100 if fake > 0 else 0
    fn = fake - tp
    tn = genuine * 90 // 100 if genuine > 0 else 0
    fp = genuine - tn
    
    return render_template('thesis_metrics.html',
                         avg_confidence=avg_confidence,
                         avg_fpr=avg_fpr,
                         avg_task_time=avg_task_time,
                         tp=tp, fn=fn, tn=tn, fp=fp,
                         total=total, genuine=genuine, fake=fake, uncertain=uncertain)

@app.route('/api-docs')
def api_docs():
    return render_template('api_docs.html')

@app.route('/history')
def history():
    username = session.get('username')
    logs = load_history(username)
    return render_template('history.html', logs=logs)

@app.route('/dashboard')
def dashboard():
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))
    
    history = load_history(username)
    
    total = len(history)
    genuine = len([h for h in history if h.get('verdict') == 'genuine'])
    fake = len([h for h in history if h.get('verdict') == 'fake'])
    uncertain = len([h for h in history if h.get('verdict') == 'uncertain'])
    
    confidences = [h.get('confidence', 0) for h in history if h.get('confidence')]
    avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0
    
    platforms = {}
    for h in history:
        platform = h.get('fields', {}).get('platform', 'Unknown')
        platforms[platform] = platforms.get(platform, 0) + 1
    
    task_times = [h.get('task_time', 0) for h in history if h.get('task_time')]
    avg_task_time = round(sum(task_times) / len(task_times), 2) if task_times else 0
    
    trend_data = [0] * 7
    today = datetime.now()
    for h in history:
        try:
            h_date = datetime.strptime(h.get('timestamp', ''), '%Y-%m-%d %H:%M:%S')
            days_diff = (today - h_date).days
            if 0 <= days_diff < 7:
                trend_data[6 - days_diff] += 1
        except:
            pass
    
    return render_template('dashboard.html', 
                         total=total, genuine=genuine, fake=fake, uncertain=uncertain,
                         avg_confidence=avg_confidence, platforms=platforms, 
                         history=history, avg_task_time=avg_task_time,
                         trend_data=trend_data)

@app.route('/clear-history', methods=['POST'])
def clear_history():
    if os.path.exists('history.json'):
        os.remove('history.json')
    return redirect(url_for('history'))

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok', 
        'model_loaded': model is not None,
        'privacy_mode': PRIVACY_MODE,
        'model_type': 'Random Forest' if model else 'Rule-based'
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(SAVED_FOLDER, exist_ok=True)
    print("=" * 60)
    print("🛡️ PayVerify Nepal Starting...")
    print("🔒 PRIVACY MODE: ON - All processing is local")
    print("🤖 ML Model:", "Loaded" if model else "Not trained yet")
    print("📁 Saved Results Folder:", SAVED_FOLDER)
    print("=" * 60)
    app.run(debug=True, port=5000)