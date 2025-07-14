import io
import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status

# 의존성 및 스키마 임포트
from core.estimation_service import MassEstimationService, mass_estimation_service
from .schemas import EstimationResponse, ErrorResponse
from config.settings import settings

router = APIRouter()

# 의존성 주입 함수
def get_mass_estimation_service() -> MassEstimationService:
    return mass_estimation_service

@router.get("/pipeline-status", summary="파이프라인 상태 확인")
async def get_pipeline_status():
    """현재 파이프라인 상태를 확인합니다."""
    return {
        "yolo_model_loaded": hasattr(mass_estimation_service, 'yolo_model'),
        "midas_model_loaded": hasattr(mass_estimation_service, 'midas_model'),
        "llm_estimator_loaded": hasattr(mass_estimation_service, 'llm_estimator'),
        "feature_extractor_loaded": hasattr(mass_estimation_service, 'feature_extractor'),
        "debug_mode": settings.DEBUG_MODE,
        "multimodal_enabled": settings.ENABLE_MULTIMODAL,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL_NAME
    }

@router.post(
    "/estimate",
    response_model=EstimationResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
    },
    summary="음식 질량 추정",
    description="이미지 파일을 업로드하여 음식의 질량을 추정합니다.",
)
async def estimate_mass(
    file: UploadFile = File(..., description="질량을 추정할 이미지 파일 (JPG, PNG 등)"),
    service: MassEstimationService = Depends(get_mass_estimation_service),
):
    """
    이미지 파일을 받아 질량 추정 파이프라인을 실행하고 결과를 반환합니다.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="업로드된 파일이 이미지 형식이 아닙니다.",
        )

    try:
        # 비동기적으로 파일 읽기
        contents = await file.read()
        # 바이트 데이터를 numpy 배열로 변환
        nparr = np.frombuffer(contents, np.uint8)
        # numpy 배열을 OpenCV 이미지로 디코딩
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="이미지 파일을 처리할 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식일 수 있습니다.",
            )

        # 서비스 실행
        result = service.run_pipeline(image, image_path=file.filename)

        if "error" in result:
             raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )

        # numpy 타입을 Python 기본 타입으로 변환
        def convert_numpy_types(obj):
            """numpy 타입을 Python 기본 타입으로 변환"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # mass_estimation 결과에서도 numpy 타입 변환
        mass_estimation = convert_numpy_types(result.get("mass_estimation", {}))

        # 간소화된 응답 생성
        detected_objects = {
            "food": len(result.get("features", {}).get("food_objects", [])),
            "reference_objects": len(result.get("features", {}).get("reference_objects", []))
        }
        
        # 질량 추정 결과 간소화
        simplified_mass_estimation = {}
        
        if "food_verifications" in mass_estimation and mass_estimation["food_verifications"]:
            # 멀티모달 검증 결과가 있는 경우
            food_ver = mass_estimation["food_verifications"][0]
            simplified_mass_estimation = {
                "estimated_mass_g": food_ver.get("verified_mass_g"),
                "confidence": food_ver.get("confidence"),
                "food_name": food_ver.get("food_name"),
                "verification_method": food_ver.get("verification_method")
            }
        elif "food_estimations" in mass_estimation and mass_estimation["food_estimations"]:
            # 초기 추정 결과가 있는 경우
            food_est = mass_estimation["food_estimations"][0]
            simplified_mass_estimation = {
                "estimated_mass_g": food_est.get("estimated_mass_g"),
                "confidence": food_est.get("confidence"),
                "food_name": food_est.get("food_name", "알수없음"),
                "verification_method": "initial_estimation"
            }
        else:
            # 기존 단일 결과 (하위 호환성)
            simplified_mass_estimation = {
                "estimated_mass_g": mass_estimation.get("estimated_mass_g"),
                "confidence": mass_estimation.get("confidence"),
                "food_name": "알수없음",
                "verification_method": "basic_estimation"
            }

        # 디버그 모드에서 간소화된 결과 로깅
        if settings.DEBUG_MODE:
            logging.info("=== 간소화된 API 응답 ===")
            logging.info(f"detected_objects: {detected_objects}")
            logging.info(f"mass_estimation: {simplified_mass_estimation}")
            logging.info("========================")

        return EstimationResponse(
            filename=file.filename,
            detected_objects=detected_objects,
            mass_estimation=simplified_mass_estimation,
        )

    except Exception as e:
        # 예기치 못한 모든 예외 처리
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"이미지 처리 중 예상치 못한 오류가 발생했습니다: {e}",
        ) 