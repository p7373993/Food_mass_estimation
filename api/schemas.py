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
    detected_objects: Dict[str, int] = Field(description="감지된 객체 개수")
    mass_estimation: Dict[str, Any] = Field(description="질량 추정 결과")

class ErrorResponse(BaseModel):
    """오류 발생 시 API 응답 모델"""
    detail: str

class HealthCheckResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str = "ok" 

# === 비동기 작업 상태 추적용 스키마 추가 ===

class TaskStatus(BaseModel):
    """비동기 작업 상태 모델"""
    task_id: str = Field(description="작업 ID")
    status: str = Field(description="작업 상태: pending, processing, completed, failed")
    progress: Optional[float] = Field(None, description="진행률 (0.0 ~ 1.0)")
    message: Optional[str] = Field(None, description="상태 메시지")
    created_at: Optional[str] = Field(None, description="작업 생성 시간")
    completed_at: Optional[str] = Field(None, description="작업 완료 시간")
    result: Optional[EstimationResponse] = Field(None, description="완료된 결과")
    error: Optional[str] = Field(None, description="오류 메시지")

class TaskCreateResponse(BaseModel):
    """비동기 작업 생성 응답"""
    task_id: str = Field(description="생성된 작업 ID")
    status: str = Field(description="초기 상태")
    message: str = Field(description="작업 시작 메시지") 