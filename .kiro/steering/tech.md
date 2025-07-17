# Technology Stack

## Core Technologies

### AI/ML Framework
- **PyTorch** - Deep learning framework for YOLO and MiDaS models
- **Ultralytics YOLO** - Object detection and segmentation
- **MiDaS** - Monocular depth estimation
- **OpenCV** - Computer vision operations
- **scikit-image** - Image processing utilities

### LLM Integration
- **Google Gemini API** - Primary LLM provider (gemini-2.5-flash)
- **OpenAI API** - Alternative LLM provider
- **google-generativeai** - Gemini Python client
- **openai** - OpenAI Python client

### Web Framework
- **FastAPI** - Modern async web framework
- **Uvicorn** - ASGI server
- **WebSockets** - Real-time communication
- **Pydantic** - Data validation and settings management

### Development Tools
- **Python 3.12+** - Primary language
- **uv** - Fast Python package manager
- **Docker** - Containerization
- **Git LFS** - Large file storage for model weights

## Build System

### Package Management
```bash
# Install dependencies
uv pip install -r requirements.txt
# or
pip install -r requirements.txt

# Development dependencies
uv pip install -e ".[dev]"
```

### Common Commands

#### Development
```bash
# Run main application
python main.py data/test1.jpg --debug

# Start API server (development)
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8001

# Run with specific model
python main.py image.jpg --model gemini-1.5-pro
```

#### Docker
```bash
# Build image
./scripts/docker-build.sh build

# Run container
./scripts/docker-build.sh run

# Development mode
./scripts/docker-build.sh dev

# Check logs
./scripts/docker-build.sh logs
```

#### Testing
```bash
# API documentation
http://localhost:8001/docs

# WebSocket test
# Open websocket_test.html in browser
```

## Configuration

### Environment Variables
- **GEMINI_API_KEY** - Google Gemini API key (required)
- **OPENAI_API_KEY** - OpenAI API key (optional)
- **LLM_PROVIDER** - "gemini" or "openai"
- **DEBUG_MODE** - Enable debug logging
- **ENABLE_MULTIMODAL** - Enable multimodal verification

### Model Files
- **weights/yolo_food_v1.pt** - Custom YOLO model (Git LFS)
- Models auto-download on first use (MiDaS, etc.)

## Performance Considerations
- **GPU Support** - CUDA acceleration when available
- **Memory Usage** - 8GB+ RAM recommended
- **Image Size** - Max 1920px for optimal performance
- **Concurrent Requests** - Limited by server resources