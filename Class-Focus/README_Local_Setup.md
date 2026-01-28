# ClassFocus - Local Windows Setup

## Quick Setup for Low-End Hardware

### Prerequisites
- Python 3.8-3.11 installed on Windows
- At least 4GB RAM
- Webcam connected

### Installation Steps

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements_local.txt
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

5. **Access the app:**
   Open your browser and go to: `http://localhost:5000`

## Performance Optimizations

This version is optimized for low-end hardware:

- **Lightweight processing:** Images are resized to 224x224 for faster analysis
- **CPU-only TensorFlow:** Uses tensorflow-cpu for better compatibility
- **Reduced analysis frequency:** Analyzes frames every 1.5 seconds instead of 1 second
- **Model caching:** Emotion detection model is loaded once at startup
- **Memory optimization:** Reduced logging and optimized OpenCV operations

## Features Included

✅ Real-time emotion detection (Happy, Sad, Neutral)  
✅ Attention monitoring with intensity levels  
✅ Face bounding box visualization  
✅ Eye closure detection for sleeping alerts  
✅ Session statistics and reporting  
✅ Smooth UI with processing feedback  

## Troubleshooting

**If the app is slow:**
- Close other applications to free up RAM
- Ensure good lighting for better face detection
- Check that your webcam is working properly

**If you get import errors:**
- Make sure the virtual environment is activated
- Try: `pip install --upgrade pip` then reinstall requirements

**If face detection isn't working:**
- Ensure good lighting
- Position your face clearly in the webcam view
- Check webcam permissions in your browser