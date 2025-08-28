import os
import base64
import cv2
import numpy as np
import logging
from collections import deque

# Configure TensorFlow environment for lightweight CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU usage

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from flask import Flask, request, jsonify, send_from_directory
from deepface import DeepFace

# Configure logging
logging.basicConfig(level=logging.INFO)  # Reduce logging for performance
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.secret_key = os.environ.get("SESSION_SECRET", "classfocus-dev-key")

# Global variable to cache the emotion detection model
emotion_model = None

# Initialize OpenCV classifiers for face and eye detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
except Exception as e:
    logger.warning(f"Could not load OpenCV cascades: {e}")
    face_cascade = cv2.CascadeClassifier()
    eye_cascade = cv2.CascadeClassifier()

# Track eye closure status across frames for sleeping detection
eye_closure_history = deque(maxlen=20)  # Track last 20 frames for sleeping
SLEEP_THRESHOLD = 15  # Need 15 consecutive closed-eye frames to trigger sleeping (~0.5 seconds)

# Track emotion history for stability
emotion_history = deque(maxlen=10)  # Track last 10 frames for emotion stability
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for emotion acceptance

# Intensity thresholds according to user specifications
HAPPY_MED = 0.50
HAPPY_HIGH = 0.75
SAD_MED = 0.50
SAD_HIGH = 0.75
GENERAL_MED = 0.50
GENERAL_HIGH = 0.75

# Performance optimization settings
TARGET_WIDTH = 224  # Resize frame to 224x224 for faster processing
TARGET_HEIGHT = 224
ANALYSIS_SKIP_FRAMES = 2  # Process every 2nd frame for better performance

def initialize_emotion_model():
    """Initialize and cache the emotion detection model at startup"""
    global emotion_model
    try:
        # Use a lightweight model and load it once
        logger.info("Initializing emotion detection model...")
        # Perform a dummy analysis to initialize the model
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.analyze(
            dummy_img, 
            actions=['emotion'], 
            enforce_detection=False, 
            silent=True,
            detector_backend='opencv'  # Use lightweight opencv detector
        )
        emotion_model = "initialized"
        logger.info("Emotion model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize emotion model: {e}")
        emotion_model = None

def preprocess_frame(img):
    """Preprocess frame for faster analysis - resize and optimize"""
    # Resize to smaller dimensions for faster processing
    height, width = img.shape[:2]
    if width > TARGET_WIDTH or height > TARGET_HEIGHT:
        # Calculate aspect ratio and resize
        aspect_ratio = width / height
        if aspect_ratio > 1:
            new_width = TARGET_WIDTH
            new_height = int(TARGET_WIDTH / aspect_ratio)
        else:
            new_height = TARGET_HEIGHT
            new_width = int(TARGET_HEIGHT * aspect_ratio)
        
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return img

def detect_face_and_eyes(img):
    """Detect face and eyes with improved accuracy, return face coordinates and eye closure status"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with better parameters for stability
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    face_box = None
    eyes_closed = False
    
    if len(faces) > 0:
        # Use the largest face for better tracking
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        face_box = (x, y, w, h)
        
        # Extract face region for eye detection with some padding
        padding = int(0.1 * h)  # 10% padding
        roi_y1 = max(0, y + padding)
        roi_y2 = min(img.shape[0], y + h//2)  # Only upper half of face for eyes
        roi_x1 = max(0, x)
        roi_x2 = min(img.shape[1], x + w)
        
        roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Detect eyes in the upper face region with better parameters
        eyes = eye_cascade.detectMultiScale(
            roi_gray, 
            scaleFactor=1.1, 
            minNeighbors=3,
            minSize=(10, 10),
            maxSize=(w//3, h//4)
        )
        
        # More sophisticated eye closure detection
        # Consider eyes closed if we detect 0-1 eyes (should normally detect 2)
        eyes_closed = len(eyes) < 2
        
        logger.debug(f"Face detected at ({x},{y},{w},{h}), found {len(eyes)} eyes, eyes_closed: {eyes_closed}")
    else:
        logger.debug("No face detected")
    
    return face_box, eyes_closed

def update_sleep_detection(eyes_closed):
    """Update sleep detection based on eye closure history - requires 15+ consecutive frames"""
    global eye_closure_history
    
    # Add current frame's eye status to history
    eye_closure_history.append(eyes_closed)
    
    # Check if the last SLEEP_THRESHOLD frames are ALL closed eyes
    recent_frames = list(eye_closure_history)[-SLEEP_THRESHOLD:]
    is_sleeping = len(recent_frames) >= SLEEP_THRESHOLD and all(recent_frames)
    
    logger.debug(f"Eye closure last {len(recent_frames)} frames: {recent_frames[-5:] if len(recent_frames) > 5 else recent_frames}, sleeping: {is_sleeping}")
    
    return is_sleeping

def stabilize_emotion_detection(emotions_dict, dominant_emotion):
    """Apply emotion history and confidence filtering to reduce random flipping"""
    global emotion_history
    
    # Get confidence for the dominant emotion
    dominant_confidence = emotions_dict.get(dominant_emotion, 0) / 100.0
    
    # Only accept emotions with confidence >= threshold
    if dominant_confidence < CONFIDENCE_THRESHOLD:
        # Default to neutral if confidence too low
        final_emotion = 'neutral'
        final_confidence = 0.5  # Default neutral confidence
        logger.debug(f"Low confidence {dominant_confidence:.2f} for {dominant_emotion}, defaulting to neutral")
    else:
        final_emotion = dominant_emotion
        final_confidence = dominant_confidence
    
    # Add to emotion history
    emotion_history.append((final_emotion, final_confidence))
    
    # Apply majority voting on recent history
    if len(emotion_history) >= 3:  # Need at least 3 frames for stability
        recent_emotions = [emotion for emotion, conf in list(emotion_history)[-5:]]  # Last 5 frames
        
        # Count occurrences
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Get most frequent emotion
        most_frequent = max(emotion_counts.items(), key=lambda x: x[1])
        stabilized_emotion = most_frequent[0]
        
        # Use average confidence for the stabilized emotion
        matching_confidences = [conf for emotion, conf in list(emotion_history)[-5:] if emotion == stabilized_emotion]
        stabilized_confidence = sum(matching_confidences) / len(matching_confidences) if matching_confidences else final_confidence
        
        logger.debug(f"Emotion stability: recent={recent_emotions}, counts={emotion_counts}, stabilized={stabilized_emotion}")
        
        return stabilized_emotion, stabilized_confidence
    else:
        return final_emotion, final_confidence

def map_emotion_to_category(dominant_emotion):
    """Map DeepFace emotion to simplified 3-category system"""
    if dominant_emotion == 'happy':
        return 'happy'
    elif dominant_emotion == 'sad':
        return 'sad'
    elif dominant_emotion == 'neutral':
        return 'neutral'
    else:
        # angry, fear, disgust, surprise -> neutral
        return 'neutral'

def get_intensity(emotion, probability):
    """Map emotion probability to intensity bucket"""
    if emotion in ['happy', 'sad']:
        if probability < 0.25:
            return 'none'
        elif probability < 0.50:
            return 'low'
        elif probability < 0.75:
            return 'medium'
        else:
            return 'high'
    else:
        # neutral has no intensity
        return 'none'

def is_attentive(emotion, intensity):
    """Determine if person is attentive based on emotion and intensity"""
    if emotion == 'neutral':
        return True
    elif emotion in ['happy', 'sad']:
        # Attentive if intensity is low or medium, not attentive if high
        return intensity in ['low', 'medium']
    else:
        return True  # Fallback

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('static', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze emotion from base64 encoded image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image/jpeg;base64,'):
            image_data = image_data.split(',')[1]
        
        # Convert to numpy array
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                'dominant': 'neutral',
                'intensity': 'none',
                'attentive': True,
                'probs': {}
            })
        
        # Preprocess frame for better performance
        processed_img = preprocess_frame(img)
        
        # Detect face and eyes first on original image for accuracy
        face_box, eyes_closed = detect_face_and_eyes(img)
        
        # Update sleep detection based on eye closure history
        is_sleeping = update_sleep_detection(eyes_closed)
        
        # Check if model is initialized, if not return neutral quickly
        if emotion_model is None:
            logger.warning("Emotion model not initialized, returning neutral")
            return jsonify({
                'dominant': 'neutral',
                'intensity': 'none',
                'attentive': True,
                'probs': {},
                'probability': 0.5,
                'face_box': [int(coord) for coord in face_box] if face_box else None,
                'sleeping': is_sleeping
            })
        
        # Analyze with DeepFace using preprocessed smaller image
        try:
            result = DeepFace.analyze(
                processed_img, 
                actions=['emotion'], 
                enforce_detection=False, 
                silent=True,
                detector_backend='opencv'  # Use lightweight detector
            )
            
            # Handle both single face and multiple faces
            if isinstance(result, list):
                result = result[0]  # Take first face
            
            emotions = result['emotion']
            original_dominant = result['dominant_emotion'].lower()
            
            # Apply emotion stabilization with confidence filtering and majority voting
            stabilized_emotion, stabilized_confidence = stabilize_emotion_detection(emotions, original_dominant)
            
            # Map to simplified 3-category system using stabilized emotion
            mapped_emotion = map_emotion_to_category(stabilized_emotion)
            
            # Get intensity for the mapped emotion using stabilized confidence
            intensity = get_intensity(mapped_emotion, stabilized_confidence)
            
            # Determine attention status
            attentive = is_attentive(mapped_emotion, intensity)
            
            # Override emotion and attention if sleeping is detected
            if is_sleeping:
                mapped_emotion = 'neutral'
                intensity = 'none'
                attentive = False  # Not attentive when sleeping
                
            # Convert emotion probabilities to decimals
            probs = {k.lower(): v / 100.0 for k, v in emotions.items()}
            
            logger.debug(f"Original: {original_dominant}, Stabilized: {stabilized_emotion} ({stabilized_confidence:.2f}), Mapped: {mapped_emotion}, Intensity: {intensity}, Attentive: {attentive}, Sleeping: {is_sleeping}")
            
            # Convert face_box coordinates to regular Python ints for JSON serialization
            face_box_json = None
            if face_box is not None:
                face_box_json = [int(coord) for coord in face_box]
            
            return jsonify({
                'dominant': mapped_emotion,
                'intensity': intensity,
                'attentive': attentive,
                'probs': probs,
                'probability': stabilized_confidence,
                'face_box': face_box_json,
                'sleeping': is_sleeping
            })
            
        except Exception as e:
            logger.error(f"DeepFace analysis error: {e}")
            # No face detected or analysis failed - return neutral as per requirements
            # But still return face detection and sleeping status if available
            # Convert face_box coordinates to regular Python ints for JSON serialization
            face_box_json = None
            if face_box is not None:
                face_box_json = [int(coord) for coord in face_box]
            
            return jsonify({
                'dominant': 'neutral',
                'intensity': 'none',
                'attentive': not is_sleeping,  # Not attentive only if sleeping
                'probs': {},
                'face_box': face_box_json,
                'sleeping': is_sleeping
            })
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

# Initialize the emotion model at startup
initialize_emotion_model()

if __name__ == '__main__':
    import os
    # Initialize model before starting the server
    if emotion_model is None:
        initialize_emotion_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)  # Disable debug for better performance