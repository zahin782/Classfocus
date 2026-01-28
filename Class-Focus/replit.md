# ClassFocus - Real-time Attention Monitoring

## Overview

ClassFocus is a Python Flask web application that monitors student attention in real-time using webcam feeds and emotion analysis. The application captures video frames from the user's webcam, analyzes emotional states using AI, and determines whether the student is attentive based on their facial expressions. It provides live feedback and session statistics to help track attention levels during learning sessions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Technology**: Vanilla HTML5, CSS3, and JavaScript (no frameworks)
- **Structure**: Single-page application with modular JavaScript class design
- **Components**:
  - Video capture interface using WebRTC getUserMedia API
  - Real-time status overlay on video feed
  - Session statistics dashboard
  - Control buttons for start/stop functionality
- **Data Flow**: Captures webcam frames every second, converts to base64, sends to backend via POST requests

### Backend Architecture
- **Framework**: Flask web server
- **Core Components**:
  - Main application server (`main.py`)
  - Static file serving for frontend assets
  - RESTful API endpoints for emotion analysis
- **API Design**:
  - `GET /` - Serves main application interface
  - `POST /analyze` - Accepts base64 image data and returns emotion analysis
- **Image Processing Pipeline**:
  - Base64 image decoding
  - OpenCV image manipulation
  - DeepFace emotion analysis integration
  - Attention logic based on emotion probabilities

### Data Processing Logic
- **Emotion Analysis**: Uses DeepFace library for facial emotion recognition
- **Emotion Mapping**: Simplifies DeepFace's 7 emotions to 3 categories:
  - Happy → "happy"
  - Sad → "sad" 
  - Neutral → "neutral"
  - All others (angry, fear, disgust, surprise) → "neutral"
- **Attention Algorithm**:
  - Neutral emotions → Always attentive
  - Happy/Sad emotions → Attentive if intensity is "low" or "medium", not attentive if "high"
  - No face detected → Treated as neutral (attentive) so class time isn't wasted
- **Intensity Mapping** (for happy/sad only):
  - None: <25% probability
  - Low: 25-49% probability
  - Medium: 50-74% probability
  - High: ≥75% probability
  - Neutral: No intensity (always "none")

### Session Management
- **Real-time Tracking**: Maintains counters for total frames and attentive frames
- **Statistics Calculation**: Computes attention percentage and session duration
- **State Management**: Client-side session state with start/stop controls

## External Dependencies

### AI/ML Libraries
- **DeepFace (v0.0.93)**: Primary emotion analysis engine for facial recognition
- **OpenCV (opencv-python-headless v4.10.0.84)**: Image processing and computer vision operations
- **NumPy (v1.26.4)**: Numerical operations for image data manipulation

### Web Framework
- **Flask (v3.0.3)**: Core web application framework for API endpoints and static file serving

### Browser APIs
- **WebRTC getUserMedia**: Browser-native webcam access for video capture
- **Canvas API**: Client-side image processing and frame extraction

### Frontend Libraries
- **Bootstrap**: UI framework for responsive design and styling components
- **Font Awesome**: Icon library for user interface elements

### Infrastructure Requirements
- **Python Runtime**: Server-side execution environment
- **Web Browser**: Modern browser with WebRTC support for camera access
- **Camera Hardware**: Webcam or built-in camera for video capture