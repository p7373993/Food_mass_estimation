from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# 내부 모듈 임포트
from . import endpoints
from .schemas import HealthCheckResponse
from core.estimation_service import mass_estimation_service
from config.settings import settings
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작/종료 시 실행되는 lifespan 이벤트 핸들러.
    서버 시작 시 모델을 미리 로드하여 응답 속도를 최적화합니다.
    """
    print("="*50)
    
    # 로깅 설정
    if settings.DEBUG_MODE:
        logging.basicConfig(level=logging.DEBUG, format=settings.LOG_FORMAT)
        print("🔍 디버그 모드 활성화")
    else:
        logging.basicConfig(level=logging.INFO, format=settings.LOG_FORMAT)
    
    logging.info("서버 시작... 모델을 로딩합니다.")
    
    # MassEstimationService 인스턴스가 생성될 때 내부적으로 모델들이 로드됩니다.
    # 여기서 service 객체를 참조함으로써 로딩을 강제 실행합니다.
    if mass_estimation_service:
        logging.info("핵심 서비스 및 AI 모델이 성공적으로 준비되었습니다.")
    else:
        logging.error("핵심 서비스 초기화에 실패했습니다.")

    print("="*50)
    yield
    # --- 서버 종료 시 실행될 코드 ---
    logging.info("서버를 종료합니다.")


app = FastAPI(
    title="음식 질량 추정 API (Food Mass Estimation API)",
    description="이미지 속 음식의 질량을 추정하는 API입니다.",
    version=settings.PIPELINE_VERSION,
    lifespan=lifespan
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",  # Live Server
        "http://127.0.0.1:5500",  # Live Server (IP)
        "http://localhost:3000",  # 다른 개발 서버
        "http://127.0.0.1:3000",  # 다른 개발 서버 (IP)
        "*"  # 개발 환경에서는 모든 origin 허용
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# API 라우터 포함
app.include_router(endpoints.router, prefix="/api/v1", tags=["Mass Estimation"])

@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Food Mass Estimation API에 오신 것을 환영합니다. /docs 로 이동하여 API 문서를 확인하세요."}

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health Check"],
    summary="서버 상태 확인",
    description="서버가 정상적으로 실행 중인지 확인합니다.",
)
def health_check():
    return {"status": "ok"} 