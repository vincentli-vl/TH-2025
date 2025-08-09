# Echocare Family Care Monitor

This program was made for the 2025 TerraHacks competition. It is based on flask, deepface, opencv, tensorflow, numpy, and more. Echocare is a comprehensive web-based monitoring system designed to help families care for loved ones with Alzheimer's and dementia. The application provides voice interaction, face recognition, activity monitoring, and care logging capabilities.

## 🌟 Features

### **Voice Assistant**
- **Voice Recognition**: Speak naturally to interact with the system

### **Face Recognition & Emotion Detection**
- **Identity Recognition**: Automatically identifies family members
- **Emotion Analysis**: Detects and tracks emotional states
- **Real-time Monitoring**: Live camera feeds with emotion overlays
- **Visitor Tracking**: Logs interactions and visitors

### **Care Management**
- **Activity Monitoring**: Track daily routines and behaviors
- **Care Logs**: Record important events and observations
- **Medication Reminders**: Schedule and track medication times
- **Episode Logging**: Document confusion events and triggers

### **Family Dashboard**
- **Real-time Status**: Live system status and alerts
- **Recent Activity**: Quick overview of recent events
- **Emergency Contact**: One-click emergency contact button
- **Responsive Design**: Works on desktop and mobile devices

## demo images
<div>
   <img src="https://github.com/vincentli-vl/TH-2025/blob/main/demo_images/landing%20page.jpg" alt="landing page" width="250" height="500"/>
   <img src="https://github.com/vincentli-vl/TH-2025/blob/main/demo_images/dashboard.jpg" alt="dashboard page" width="250" height="500"/>
   <img src="https://github.com/vincentli-vl/TH-2025/blob/main/demo_images/echocare.jpg" alt="echocare page" width="250" height="500"/>
   <img src="https://github.com/vincentli-vl/TH-2025/blob/main/demo_images/camera.jpg" alt="camera page" width="250" height="500"/>
   <img src="https://github.com/vincentli-vl/TH-2025/blob/main/demo_images/logger.jpg" alt="logger page" width="250" height="500"/>
   <img src="https://github.com/vincentli-vl/TH-2025/blob/main/demo_images/logger%202.jpg" alt="logger page" width="250" height="500"/>
   <img src="https://github.com/vincentli-vl/TH-2025/blob/main/demo_images/logger%203.jpg" alt="logger page" width="250" height="500"/>
   <img src="https://github.com/vincentli-vl/TH-2025/blob/main/demo_images/profile.jpg" alt="profile page" width="250" height="500"/>
   <img src="https://github.com/vincentli-vl/TH-2025/blob/main/demo_images/settings.jpg" alt="settings page" width="250" height="500"/>
</div>

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Webcam** (for face recognition features)
- **Microphone** (for voice interaction)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd TH-2025
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:5000`
   - The application will start with the landing page

3. **Access different features**
   - **Dashboard**: `http://localhost:5000/dashboard`
   - **Voice Assistant**: `http://localhost:5000/voice`
   - **Live Camera**: `http://localhost:5000/camera`
   - **Care Logs**: `http://localhost:5000/journal`
   - **Profile**: `http://localhost:5000/profile`
   - **Settings**: `http://localhost:5000/settings`

## 📁 Project Structure

```
TH-2025/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Landing page
│   ├── dashboard.html    # Main dashboard
│   ├── voice.html        # Voice assistant interface
│   ├── camera.html       # Camera monitoring
│   ├── journal.html      # Care logs
│   ├── profile.html      # User profile
│   └── settings.html     # Application settings
├── static/               # Static assets
│   ├── images/           # Images and logos
│   ├── audio/            # Audio response files
│   └── styles.css        # Custom styles
├── conversations/        # Conversation history (auto-created)
└── emojis/              # Emotion emoji files (optional)
```

## 🎯 Usage Guide

### **Voice Assistant**
1. Navigate to the Voice Assistant page
2. Click the "Record" button or press Spacebar
3. Speak your message clearly
4. The system will respond

### **Face Recognition**
1. Ensure webcam is connected and accessible
2. Add reference photos to the root directory
3. Access the camera page to see live emotion detection
4. System will automatically identify known family members

### **Care Logging**
1. Use the Care Logs page to record important events
2. Log confusion episodes, medication times, and observations
3. Track patterns and triggers over time
4. Export logs for healthcare providers

### **Dashboard Monitoring**
1. View real-time system status
2. Check recent alerts and activities
3. Access emergency contact features
4. Monitor overall care system health

## 🔧 Configuration

### **Face Recognition**
- Add high-quality photos of family members
- Use clear, front-facing images
- Multiple photos per person improve recognition accuracy

## 🛠️ Troubleshooting

### **Common Issues**

**Voice Recognition Not Working**
- Ensure microphone permissions are granted
- Check browser compatibility (Chrome/Edge recommended)
- Try text input fallback if voice fails

**Camera Not Accessible**
- Check webcam permissions
- Ensure no other applications are using the camera
- Try refreshing the page

**Face Recognition Errors**
- Verify reference photos are in the root directory
- Check photo quality and format
- Ensure photos are clear and well-lit

### **Error Messages**

- **"Camera initialization error"**: Check webcam connection and permissions
- **"Error loading reference embeddings"**: Missing or invalid reference photos

## 🔒 Privacy & Security

- **Local Processing**: All face recognition and voice processing happens locally
- **No Cloud Storage**: Conversations and data stay on your device
- **Secure Access**: No external data transmission
- **Family Control**: Complete control over data and settings

---

**Echocare Family Care Monitor** - Supporting families through memory challenges with technology and compassion.
