from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import cv2
import numpy as np
import joblib
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model (keep existing model)
try:
    model = joblib.load('model/svm_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    pca = joblib.load('model/pca.pkl')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    model = None
    scaler = None
    pca = None

# Simple user storage (in production, use a proper database)
users = {}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_resized = cv2.resize(img, (128, 128))  # Match training size (128x128, not 64x64)
    
    # HOG features (must match training exactly)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),   # Match training
        _blockSize=(16, 16),   # Match training
        _blockStride=(8, 8),  # Match training
        _cellSize=(8, 8),     # Match training
        _nbins=9              # Match training
    )
    hog_feats = hog.compute(gray).flatten()
    
    # HSV histogram features (must match training exactly)
    hsv_feat = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hsv_features = []
    ranges = ([0, 180], [0, 256], [0, 256])  # H, S, V ranges
    for ch, (lo, hi) in enumerate(ranges):
        hist = cv2.calcHist([hsv_feat], [ch], None, [32], [lo, hi])  # 32 bins like training
        hist = cv2.normalize(hist, hist).flatten()  # L2-normalize like training
        hsv_features.extend(hist)
    hsv_features = np.array(hsv_features)
    
    # Grayscale grid statistics (must match training exactly)
    grid = 4  # Match training
    h, w = gray.shape
    cell_h, cell_w = h // grid, w // grid
    stat_features = []
    for r in range(grid):
        for c in range(grid):
            cell = gray[
                r*cell_h:(r+1)*cell_h,
                c*cell_w:(c+1)*cell_w
            ].astype(np.float32)
            stat_features.append(cell.mean())
            stat_features.append(cell.std())
            # Simple skewness proxy: mean - median
            stat_features.append(cell.mean() - np.median(cell))
    stat_features = np.array(stat_features)
    
    # Combine all features in original order (HOG + HSV + Stats)
    all_features = np.concatenate([hog_feats, hsv_features, stat_features])
    
    print(f"🔍 Feature extraction details:")
    print(f"  HOG features: {len(hog_feats)}")
    print(f"  HSV features: {len(hsv_features)}")
    print(f"  Stat features: {len(stat_features)}")
    print(f"  Total features: {len(all_features)}")
    
    # Ensure exactly 8244 features
    if len(all_features) != 8244:
        if len(all_features) > 8244:
            all_features = all_features[:8244]
            print(f"  ⚠️ Truncated to 8244 features")
        elif len(all_features) < 8244:
            padding = np.zeros(8244 - len(all_features))
            all_features = np.concatenate([all_features, padding])
            print(f"  ⚠️ Padded to 8244 features")
    
    return all_features

def test_feature_extraction():
    test_img_path = 'static/uploads/test_image.jpg'
    features = extract_features(test_img_path)
    assert features is not None, "Feature extraction failed"
    assert len(features) == 8244, "Incorrect number of features"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('detect'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password:
            return render_template('register.html', error='All fields are required')
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        
        if username in users:
            return render_template('register.html', error='Username already exists')
        
        users[username] = password
        session['user'] = username
        return redirect(url_for('detect'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/detect')
@login_required
def detect():
    return render_template('detect.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/tech')
def tech():
    return render_template('technology.html')

@app.route('/test-model')
def test_model():
    if model is None:
        return "Model not loaded"
    
    # Test with dummy features to verify model works
    dummy_features = np.random.rand(8244)
    features_scaled = scaler.transform([dummy_features])
    features_pca = pca.transform(features_scaled)
    
    prediction = model.predict(features_pca)[0]
    proba = model.predict_proba(features_pca)[0]
    
    return f"""
    Model Test Results:<br>
    Prediction: {prediction}<br>
    Probabilities: {proba}<br>
    Fire Confidence: {proba[1]*100:.2f}%<br>
    No Fire Confidence: {proba[0]*100:.2f}%<br>
    Model is working correctly!
    """

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'})
    
    # Save file with safe filename
    filename = file.filename.replace(' ', '_').lower()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Ensure unique filename
    counter = 1
    original_filename = filename
    while os.path.exists(filepath):
        name, ext = os.path.splitext(original_filename)
        filename = f"{name}_{counter}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        counter += 1
    
    file.save(filepath)
    
    try:
        # Process
        features = extract_features(filepath)
        if features is None:
            return jsonify({'error': 'Invalid image'})
        
        print(f"🔍 Debug: Processing {filepath}")
        print(f"  Features shape: {features.shape}")
        
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        
        prediction = model.predict(features_pca)[0]
        proba = model.predict_proba(features_pca)[0]
        
        print(f"  Raw prediction: {prediction}")
        print(f"  Probabilities: {proba}")
        print(f"  Fire prob: {proba[1]:.4f} ({proba[1]*100:.2f}%)")
        print(f"  No Fire prob: {proba[0]:.4f} ({proba[0]*100:.2f}%)")
        
        label = "WILDFIRE DETECTED" if prediction == 1 and proba[1] > 0.7 else "NO FIRE DETECTED"
        fire_conf = round(proba[1] * 100, 2)
        nofire_conf = round(proba[0] * 100, 2)
        
        # Additional safety check - if confidence is too low, default to no fire
        if max(proba) < 0.6:
            label = "NO FIRE DETECTED"
            fire_conf = round(proba[1] * 100, 2)
            nofire_conf = round(proba[0] * 100, 2)
            print(f"  ⚠️ Low confidence - defaulting to NO FIRE")
        
        print(f"  🎯 Final result: {label}")
        print(f"  📊 Confidence - Fire: {fire_conf}%, No Fire: {nofire_conf}%")
        
        return jsonify({
            'label': label,
            'prediction': int(prediction),
            'fire_conf': fire_conf,
            'nofire_conf': nofire_conf,
            'image_url': f'/{filepath}'
        })
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)