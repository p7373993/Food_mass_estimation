import os
from typing import Dict, Any
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """파이프라인 설정 클래스"""
    
    # 모델 설정
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "best_model_food_only.pt")
    MIDAS_MODEL_TYPE = os.getenv("MIDAS_MODEL_TYPE", "DPT_Large")  # "DPT_Large", "DPT_Hybrid", "MiDaS_small"
    
    # LLM 설정
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "openai"
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")
    MULTIMODAL_MODEL_NAME = os.getenv("MULTIMODAL_MODEL_NAME", "gemini-1.5-flash")
    
    # API 키 설정
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # 파이프라인 설정
    ENABLE_MULTIMODAL = os.getenv("ENABLE_MULTIMODAL", "true").lower() == "true"
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    SIMPLE_DEBUG = os.getenv("SIMPLE_DEBUG", "false").lower() == "true"
    SAVE_RESULTS = os.getenv("SAVE_RESULTS", "true").lower() == "true"
    
    # 이미지 처리 설정
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1920"))  # 최대 이미지 크기 (픽셀)
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))  # YOLO 신뢰도 임계값
    
    # 파일 경로 설정
    RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
    LOGS_DIR = os.getenv("LOGS_DIR", "logs")
    DATA_DIR = os.getenv("DATA_DIR", "data")
    
    # 로깅 설정
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # 질량 추정 설정
    DEFAULT_FOOD_DENSITY = float(os.getenv("DEFAULT_FOOD_DENSITY", "0.7"))  # g/cm³
    DEFAULT_PORTION_SIZE = int(os.getenv("DEFAULT_PORTION_SIZE", "100"))  # g
    DEFAULT_TYPICAL_PORTION = int(os.getenv("DEFAULT_TYPICAL_PORTION", "100"))  # g
    DEFAULT_PIXEL_PER_CM = float(os.getenv("DEFAULT_PIXEL_PER_CM", "100.0"))
    DEFAULT_DEPTH_VARIATION = float(os.getenv("DEFAULT_DEPTH_VARIATION", "1.0"))
    DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE", "0.3"))
    DEFAULT_MASS = float(os.getenv("DEFAULT_MASS", "100.0"))
    
    # 모델 입력 크기
    MIDAS_INPUT_SIZE = int(os.getenv("MIDAS_INPUT_SIZE", "384"))
    
    # 파이프라인 설정
    PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "1.0.0")
    
    # 계산 설정
    MINIMUM_CONFIDENCE_THRESHOLD = float(os.getenv("MINIMUM_CONFIDENCE_THRESHOLD", "0.3"))
    MAXIMUM_CONFIDENCE_THRESHOLD = float(os.getenv("MAXIMUM_CONFIDENCE_THRESHOLD", "1.0"))
    VOLUME_CALCULATION_LAYERS = int(os.getenv("VOLUME_CALCULATION_LAYERS", "10"))
    SCALE_FACTOR_CONFIDENCE_DIVISOR = float(os.getenv("SCALE_FACTOR_CONFIDENCE_DIVISOR", "3.0"))
    
    # 기준 물체 설정
    REFERENCE_OBJECTS_DB_PATH = os.getenv("REFERENCE_OBJECTS_DB_PATH", "data/reference_objects.json")
    
    # 성능 설정
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """모든 설정을 딕셔너리로 반환"""
        return {
            attr: getattr(cls, attr) 
            for attr in dir(cls) 
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }
    
    @classmethod
    def update_from_dict(cls, settings: Dict[str, Any]):
        """딕셔너리로부터 설정 업데이트"""
        for key, value in settings.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
    
    @classmethod
    def validate_settings(cls) -> Dict[str, str]:
        """설정 유효성 검사"""
        errors = {}
        
        # API 키 확인
        if cls.LLM_PROVIDER == "gemini":
            if not cls.GEMINI_API_KEY:
                errors["GEMINI_API_KEY"] = "Gemini API 키가 설정되지 않았습니다"
        elif cls.LLM_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                errors["OPENAI_API_KEY"] = "OpenAI API 키가 설정되지 않았습니다"
        else:
            errors["LLM_PROVIDER"] = f"지원되지 않는 LLM 제공자: {cls.LLM_PROVIDER}"
        
        # 모델 파일 확인
        if not os.path.exists(cls.YOLO_MODEL_PATH):
            errors["YOLO_MODEL_PATH"] = f"YOLO 모델 파일이 존재하지 않습니다: {cls.YOLO_MODEL_PATH}"
        
        # 디렉토리 생성
        for dir_path in [cls.RESULTS_DIR, cls.LOGS_DIR, cls.DATA_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        return errors 