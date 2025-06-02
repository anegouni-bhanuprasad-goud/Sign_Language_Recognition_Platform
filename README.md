# Sign Language Recognition System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-red.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time AI-powered sign language recognition system that translates sign language gestures into text using computer vision and deep learning. Built as a major project by students of Mahaveer Institute of Science and Technology.


## 🌟 Features

### 🎥 **Real-Time Recognition**
- Live camera processing at 20 FPS
- Instant sign language detection and translation
- Sub-second response time
- Continuous recognition capabilities

### 🧠 **AI-Powered Accuracy**
- Custom deep neural network architecture
- 98% recognition accuracy
- Trained on diverse sign language datasets
- Context-aware predictions

### 👋 **Holistic Detection**
- **Hand Landmarks**: 21 points per hand for precise gesture tracking
- **Facial Landmarks**: 468 points for facial expression analysis
- **Pose Landmarks**: 33 points for body posture understanding
- Complete gesture interpretation

### 🌐 **Dual Input Modes**
- **Live Camera Mode**: Real-time webcam processing
- **Video Upload Mode**: Analyze pre-recorded sign language videos
- Seamless switching between modes
- File validation and size limits (50MB max)

### 🔒 **Privacy & Security**
- Local video processing
- No video data storage
- Secure HTTPS connections
- Privacy-first design

### 📱 **Cross-Platform Compatibility**
- Works in all modern browsers (Chrome, Firefox, Safari, Edge)
- Responsive design for desktop, tablet, and mobile
- No app installation required
- Cross-platform support (Windows, macOS, Linux)

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   AI Model     │
│   (HTML/CSS/JS) │◄──►│   (FastAPI)      │◄──►│   (TensorFlow)  │
│                 │    │                  │    │                 │
│ • Camera Feed   │    │ • API Endpoints  │    │ • Neural Net    │
│ • MediaPipe     │    │ • Video Upload   │    │ • Landmark      │
│ • UI Controls   │    │ • Database       │    │   Processing    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Webcam (for live recognition)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anegouni-bhanuprasad-goud/Sign_Language_Recognition_Platform.git
   cd Sign_Language_Recognition_Platform
   ```

2. **Create virtual environment**
   ```bash
   python -m venv iven
   source iven/bin/activate  # On Windows: iven\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Open your browser**
   Navigate to `http://127.0.0.1:8000`

## 💻 Usage

### Live Camera Mode
1. Click **"Live Camera"** tab
2. Click **"Start Camera"** to initialize webcam and canvas
3. Click **"Start Prediction"** to begin recognition
4. Perform sign language gestures in front of the camera
5. View real-time predictions and history

### Video Upload Mode
1. Click **"Upload Video"** tab
2. Select a video file (max 50MB)
3. Click **"Analyze Video"** to process
4. View prediction results

### Supported Gestures
The system currently recognizes the following sign language words:
- Hello
- Thank You
- Please
- Sorry
- Yes
- No
- Good
- Bad
- Help
- More
- *(Additional gestures can be added through model retraining)*

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main application interface |
| `POST` | `/predict` | Live camera prediction |
| `POST` | `/predict-video` | Video upload prediction |
| `GET` | `/all_records` | Retrieve all prediction history |
| `GET` | `/record?id={id}` | Get specific prediction record |
| `DELETE` | `/delete_record?id={id}` | Delete prediction record |
| `GET` | `/reset` | Reset database |

### Request/Response Examples

#### Live Prediction
```json
POST /predict
{
  "frame_data": [
    {
      "frame_id": 0,
      "face_landmarks": [...],
      "pose_landmarks": [...],
      "left_hand_landmarks": [...],
      "right_hand_landmarks": [...]
    }
  ]
}
```

#### Response
```json
{
  "prediction": "Model Predicted : Hello"
}
```

## 🏗️ Project Structure

```
Sign_Language_Recognition_Platform/
├── 📁 static/                      # Frontend assets
│   ├── 📄 index.html              # Main application page
│   ├── 📄 features.html           # Features showcase
│   ├── 📄 about.html              # About page
│   ├── 📄 script.js               # JavaScript functionality
│   ├── 📄 style.css               # Main styles
│   └── 📄 styles.css              # Additional styles
├── 📁 __pycache__/                # Python cache files
├── 📁 iven/                       # Virtual environment
├── 📄 main.py                     # FastAPI application
├── 📄 predictions.py              # ML model inference
├── 📄 models.py                   # Database models
├── 📄 database.py                 # Database configuration
├── 📄 BaseModels.py               # Pydantic models
├── 📄 requirements.txt            # Dependencies
├── 📄 holistic-custom-*.h5        # Trained model file
├── 📄 database.db                 # SQLite database
├── 📄 Procfile                    # Deployment config
├── 📄 runtime.txt                 # Python version
└── 📄 README.md                   # Project documentation
```

## 🔬 Technical Implementation

### Machine Learning Pipeline
1. **Data Collection**: MediaPipe extracts landmarks from video frames
2. **Preprocessing**: Landmark normalization and sequence padding
3. **Model Architecture**: Custom CNN-LSTM hybrid network
4. **Training**: Multi-class classification with categorical crossentropy
5. **Inference**: Real-time prediction on 60-frame sequences

### Frontend Technology Stack
- **HTML5/CSS3**: Modern responsive design
- **JavaScript ES6+**: Interactive functionality
- **MediaPipe**: Real-time landmark detection
- **Canvas API**: Video processing and visualization

### Backend Technology Stack
- **FastAPI**: High-performance web framework
- **SQLAlchemy**: Database ORM
- **SQLite**: Lightweight database
- **Uvicorn**: ASGI server
- **TensorFlow**: Deep learning framework

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 98.0% |
| **Model Size** | ~15MB |
| **Inference Time** | <2 seconds |
| **Training Data** | 10+ sign classes |
| **Frame Processing** | 20 FPS |

## 🚀 Deployment

### Local Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
The application includes configuration for deployment on platforms like:
- **Heroku**: `Procfile` and `runtime.txt` included
- **Docker**: Can be containerized
- **AWS/GCP**: Cloud-ready architecture

## 👥 Team

**Major Project Team - Mahaveer Institute of Science and Technology**

| Name | Student ID | Role |
|------|------------|------|
| **Anegouni Bhanu Prasad Goud** | 21E31A6604 | Team Leader & Full Stack Developer |
| **Bodhu Harshitha** | 21E31A6609 | ML Engineer & Backend Developer |
| **Eda Srinath** | 21E31A6612 | Frontend Developer & UI/UX Designer |
| **Megadi Vinay Kumar** | 21E31A6620 | Data Scientist & Model Optimization |

**Institution**: Mahaveer Institute of Science and Technology  
**Department**: Computer Science & Engineering (AI & ML)  
**Academic Year**: 2024-25

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 coding standards
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe Team** for the holistic landmark detection framework
- **TensorFlow Team** for the machine learning platform
- **FastAPI Team** for the modern web framework
- **Mahaveer Institute of Science and Technology** for academic support
- **Open Source Community** for inspiration and resources

## 📞 Contact

- **Project Team**: [bhanuprasadgoudanegouni@gmail.com](mailto:bhanuprasadgoudanegouni@gmail.com)
- **GitHub Repository**: [Sign_Language_Recognition_Platform](https://github.com/anegouni-bhanuprasad-goud/Sign_Language_Recognition_Platform)
- **Project Report**: [Available on request]

## 🔮 Future Enhancements

- [ ] Support for more sign language gestures
- [ ] Multiple sign language variants (ASL, BSL, ISL)
- [ ] Mobile app development
- [ ] Real-time translation to speech
- [ ] Multi-user collaboration features
- [ ] Advanced analytics dashboard
- [ ] API for third-party integrations

---

<div align="center">

**Made with ❤️ by MIST Students**

[⭐ Star this repo](https://github.com/anegouni-bhanuprasad-goud/Sign_Language_Recognition_Platform) | [🐛 Report Bug](https://github.com/anegouni-bhanuprasad-goud/Sign_Language_Recognition_Platform/issues) | [✨ Request Feature](https://github.com/anegouni-bhanuprasad-goud/Sign_Language_Recognition_Platform/issues)

</div>
