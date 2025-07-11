from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import warnings

# 외부 라이브러리 워닝 필터링
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# --- .env 파일 명시적 로드 ---
# 프로젝트 루트에 있는 .env 파일의 절대 경로를 찾아 명시적으로 로드합니다.
# 이렇게 하면 uvicorn 실행 위치와 관계없이 항상 올바른 .env를 사용하게 됩니다.
# override=True 옵션은 시스템 환경변수보다 .env 파일의 값을 우선적으로 적용시킵니다.
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"✅ .env 파일을 명시적으로 로드했습니다: {dotenv_path}")
else:
    print(f"⚠️ .env 파일이 존재하지 않아 로드하지 않았습니다. 환경 변수를 직접 사용합니다.")


class Settings(BaseSettings):
    """
    애플리케이션 설정을 관리하는 클래스.
    .env 파일 또는 환경 변수에서 설정을 로드합니다.
    """
    # python-dotenv로 환경변수를 미리 로드했으므로,
    # pydantic-settings는 자동으로 환경변수에서 설정을 읽어옵니다.
    model_config = SettingsConfigDict(extra='ignore')

    # 프로젝트 루트 디렉토리
    ROOT_DIR: Path = Path(__file__).parent.parent

    # 모델 설정
    YOLO_MODEL_PATH: Path = ROOT_DIR / "weights/yolo_food_v1.pt"
    MIDAS_MODEL_TYPE: str = "DPT_Large"
    
    # LLM 설정
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL_NAME: str = "gemini-1.5-flash"
    MULTIMODAL_MODEL_NAME: str = "gemini-1.5-flash"
    
    # API 키 설정
    GEMINI_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    
    # 파이프라인 설정
    PIPELINE_VERSION: str = "1.0.0"
    ENABLE_MULTIMODAL: bool = True
    DEBUG_MODE: bool = False
    SIMPLE_DEBUG: bool = False
    SAVE_RESULTS: bool = True
    
    # 이미지 처리 설정
    MAX_IMAGE_SIZE: int = 1920
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # 파일 경로 설정
    RESULTS_DIR: Path = ROOT_DIR / "results"
    LOGS_DIR: Path = ROOT_DIR / "logs"
    DATA_DIR: Path = ROOT_DIR / "data"
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 질량 추정 설정
    DEFAULT_FOOD_DENSITY: float = 0.7
    DEFAULT_PORTION_SIZE: int = 100
    DEFAULT_TYPICAL_PORTION: int = 100
    DEFAULT_PIXEL_PER_CM: float = 100.0
    DEFAULT_DEPTH_VARIATION: float = 1.0
    DEFAULT_CONFIDENCE: float = 0.3
    DEFAULT_MASS: float = 100.0
    
    # 모델 입력 크기
    MIDAS_INPUT_SIZE: int = 384
    
    # 계산 설정
    MINIMUM_CONFIDENCE_THRESHOLD: float = 0.3
    MAXIMUM_CONFIDENCE_THRESHOLD: float = 1.0
    VOLUME_CALCULATION_LAYERS: int = 10
    SCALE_FACTOR_CONFIDENCE_DIVISOR: float = 3.0
    
    # 픽셀-실제 크기 변환 설정
    DEFAULT_PIXEL_TO_MM: float = 0.5
    DEFAULT_PIXEL_TO_CM: float = 0.05
    
    # 부피 계산 설정
    MIN_VOLUME_CM3: float = 5.0
    MAX_VOLUME_CM3: float = 1000.0
    FALLBACK_MIN_VOLUME_CM3: float = 20.0
    FALLBACK_MAX_VOLUME_CM3: float = 500.0
    
    # 형태 보정 계수
    SHAPE_FACTOR_CIRCULAR: float = 0.65
    SHAPE_FACTOR_ELLIPTICAL: float = 0.60
    SHAPE_FACTOR_IRREGULAR: float = 0.55
    SHAPE_FACTOR_DEFAULT: float = 0.6
    
    # 기준 물체 기본 크기 (이어폰 케이스)
    DEFAULT_REFERENCE_AREA_CM2: float = 25.0
    DEFAULT_REFERENCE_THICKNESS_CM: float = 2.5
    DEFAULT_REFERENCE_VOLUME_CM3: float = 62.5
    
    # 기준 물체 설정
    REFERENCE_OBJECTS_DB_PATH: Path = DATA_DIR / "reference_objects.json"
    
    # 성능 설정
    BATCH_SIZE: int = 1
    USE_GPU: bool = True

    @field_validator('YOLO_MODEL_PATH')
    @classmethod
    def yolo_path_must_exist(cls, v: Path) -> Path:
        # CI/CD 환경이나 빌드 과정에서는 파일이 없을 수 있으므로, 경고만 출력하도록 조정
        if not v.exists():
            print(f"Warning: YOLO 모델 파일이 존재하지 않습니다: {v}")
        return v

    @field_validator('REFERENCE_OBJECTS_DB_PATH')
    @classmethod
    def ref_obj_db_path_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            # 이 파일은 필수적이므로 에러 발생
            raise ValueError(f"기준 물체 DB 파일이 존재하지 않습니다: {v}")
        return v
        
# 설정 객체 인스턴스 생성
settings = Settings()

# 필요한 디렉토리 생성
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True) 