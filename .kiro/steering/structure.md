# Project Structure

## Architecture Pattern

This project follows a **layered service architecture** with clear separation of concerns:

- **Entry Points** (`main.py`, `api/`) - CLI and web interfaces
- **Core Services** (`core/`) - Business logic orchestration
- **Models** (`models/`) - AI model wrappers with singleton pattern
- **Utilities** (`utils/`) - Shared helper functions
- **Configuration** (`config/`) - Centralized settings management

## Directory Organization

### Core Application
```
├── main.py                 # CLI entry point
├── core/
│   └── estimation_service.py    # Main pipeline orchestrator
├── config/
│   └── settings.py         # Centralized configuration (Pydantic)
```

### AI Models (Singleton Pattern)
```
├── models/
│   ├── yolo_model.py       # YOLO segmentation wrapper
│   ├── midas_model.py      # MiDaS depth estimation wrapper
│   ├── llm_model.py        # Gemini/OpenAI LLM wrapper
│   └── __pycache__/
```

### API Layer
```
├── api/
│   ├── main.py             # FastAPI application
│   ├── endpoints.py        # API route handlers
│   ├── schemas.py          # Pydantic data models
│   └── __pycache__/
```

### Utilities
```
├── utils/
│   ├── base_model.py       # Base class for AI models
│   ├── feature_extraction.py    # Computer vision feature extraction
│   ├── camera_info_extractor.py # EXIF data processing
│   ├── reference_objects.py     # Reference object management
│   ├── debug_helper.py          # Debug visualization
│   ├── logging_utils.py         # Logging configuration
│   └── __pycache__/
```

### Data and Assets
```
├── data/                   # Input images and reference data
│   ├── *.jpg              # Sample images
│   └── reference_objects.json   # Reference object specifications
├── weights/
│   └── yolo_food_v1.pt    # Custom YOLO model (Git LFS)
├── results/               # Generated outputs (auto-created)
│   ├── *_segmentation.jpg # Visualization files
│   └── *_depth.jpg        # Depth map visualizations
└── logs/                  # Application logs (auto-created)
    └── main.log
```

### Deployment
```
├── Dockerfile             # Container definition
├── docker-compose.yml     # Development deployment
├── docker-compose.prod.yml # Production deployment
├── nginx.conf             # Reverse proxy configuration
└── scripts/
    └── docker-build.sh    # Build and deployment scripts
```

## Key Design Patterns

### Singleton Models
All AI models (`yolo_model`, `midas_model`, `llm_estimator`) use singleton pattern to avoid reloading heavy models.

### Service Layer
`MassEstimationService` orchestrates the entire pipeline, calling models in sequence and handling errors.

### Configuration Management
`settings.py` uses Pydantic for type-safe configuration with `.env` file support.

### Base Model Pattern
`BaseModel` provides common functionality for all AI model wrappers (initialization, device management, logging).

## File Naming Conventions

- **Snake_case** for Python files and variables
- **PascalCase** for class names
- **UPPER_CASE** for constants and environment variables
- **Descriptive names** that indicate purpose (e.g., `estimation_service.py`, `feature_extraction.py`)

## Import Patterns

- **Absolute imports** from project root
- **Relative imports** only within same package
- **Lazy loading** for heavy dependencies (models)
- **Central configuration** imported as `from config.settings import settings`

## Error Handling

- **Graceful degradation** - system continues with reduced functionality
- **Structured error responses** with error codes and messages
- **Comprehensive logging** at appropriate levels
- **Fallback mechanisms** for critical components