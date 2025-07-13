import io
import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status

# 의존성 및 스키마 임포트
from core.estimation_service import MassEstimationService, mass_estimation_service
from .schemas import EstimationResponse, ErrorResponse

router = APIRouter()

# 의존성 주입 함수
def get_mass_estimation_service() -> MassEstimationService:
    return mass_estimation_service

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

        return EstimationResponse(
            filename=file.filename,
            mass_estimation=result.get("mass_estimation", {}),
            features=result.get("features", {}),
        )

    except Exception as e:
        # 예기치 못한 모든 예외 처리
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"이미지 처리 중 예상치 못한 오류가 발생했습니다: {e}",
        ) 