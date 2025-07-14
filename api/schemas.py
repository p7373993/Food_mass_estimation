from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class MassEstimation(BaseModel):
    """질량 추정 결과 모델"""
    estimated_mass_g: Optional[float] = Field(None, description="추정된 질량 (그램)")
    confidence: Optional[float] = Field(None, description="추정 신뢰도 (0.0 ~ 1.0)")
    reasoning: Optional[str] = Field(None, description="추정 근거")
    error: Optional[str] = None
    
    # 여러 음식 처리 결과 (새로운 구조)
    food_count: Optional[int] = Field(None, description="감지된 음식 개수")
    food_estimations: Optional[List[Dict[str, Any]]] = Field(None, description="개별 음식 추정 결과")
    food_verifications: Optional[List[Dict[str, Any]]] = Field(None, description="멀티모달 검증 결과")

class SimplifiedFeature(BaseModel):
    """간소화된 특징 정보 모델"""
    class_name: str
    confidence: float
    pixel_area: int
    depth_info: Dict[str, float]

class EstimationResponse(BaseModel):
    """성공적인 질량 추정 API 응답 모델"""
    filename: str
    mass_estimation: MassEstimation
    features: Dict[str, Any] = Field(description="추출된 특징들의 요약 정보")

class ErrorResponse(BaseModel):
    """오류 발생 시 API 응답 모델"""
    detail: str

class HealthCheckResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str = "ok" 