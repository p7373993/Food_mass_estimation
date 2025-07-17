# Product Overview

## AI-Based Food Mass Estimation System

This is an AI-powered food mass estimation system that combines computer vision and large language models to accurately estimate food mass from images. The system uses a multi-stage pipeline:

1. **YOLO Segmentation** - Detects and segments food items and reference objects (like earphone cases)
2. **MiDaS Depth Estimation** - Generates depth maps for volume calculation
3. **Feature Extraction** - Calculates pixel areas, depth distributions, and relative sizes
4. **LLM Analysis** - Uses Gemini/OpenAI models for final mass estimation and verification

## Key Features

- **Reference Object Scaling**: Uses known-size objects for accurate scale calculation
- **Volume-based Calculation**: 3D volume estimation using depth information
- **Multi-modal Verification**: LLM verification with visual analysis
- **Real-time API**: FastAPI server with WebSocket support for live processing
- **Flexible Configuration**: Support for multiple LLM providers and model options

## Target Use Cases

- Food portion estimation for dietary tracking
- Nutritional analysis applications
- Restaurant and food service applications
- Research in computer vision and food analysis

## Languages

- Primary documentation and UI: Korean (한국어)
- Code comments and technical documentation: English
- API responses: Both Korean and English supported