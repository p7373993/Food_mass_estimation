import io
import numpy as np
import cv2
import uuid
import asyncio
import logging
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import Dict, List

# 의존성 및 스키마 임포트
from core.estimation_service import MassEstimationService, mass_estimation_service
from .schemas import EstimationResponse, ErrorResponse, TaskStatus, TaskCreateResponse
from config.settings import settings

router = APIRouter()

# 비동기 작업 저장소 (메모리 기반)
task_store = {}

# WebSocket 연결 관리자
class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)
        logging.info(f"WebSocket 연결됨: task_id={task_id}, 총 연결 수={len(self.active_connections[task_id])}")
    
    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            if websocket in self.active_connections[task_id]:
                self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        logging.info(f"WebSocket 연결 해제: task_id={task_id}")
    
    async def send_task_update(self, task_id: str, message: dict):
        """특정 작업의 모든 WebSocket 연결에 업데이트 전송"""
        if task_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[task_id]:
                try:
                    await websocket.send_text(json.dumps(message, ensure_ascii=False))
                except WebSocketDisconnect:
                    disconnected.append(websocket)
                except Exception as e:
                    logging.error(f"WebSocket 메시지 전송 실패: {e}")
                    disconnected.append(websocket)
            
            # 연결이 끊어진 WebSocket 제거
            for websocket in disconnected:
                self.disconnect(websocket, task_id)

# WebSocket 매니저 인스턴스 생성
websocket_manager = WebSocketManager()

# 스레드 풀 생성 (AI 모델 실행용)
thread_pool = ThreadPoolExecutor(max_workers=2)

# 의존성 주입 함수
def get_mass_estimation_service() -> MassEstimationService:
    return mass_estimation_service

# 동기 함수를 비동기로 실행하는 헬퍼 함수
def run_pipeline_sync(service, image, image_path):
    """동기 파이프라인을 별도 스레드에서 실행"""
    return service.run_pipeline(image, image_path)

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
    # 디버깅을 위한 파일 정보 로깅
    if settings.DEBUG_MODE:
        logging.info(f"파일 업로드 정보:")
        logging.info(f"  - filename: {file.filename}")
        logging.info(f"  - content_type: {file.content_type}")
        logging.info(f"  - size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    # 파일 타입 검증 (content_type이 None일 수 있으므로 안전하게 처리)
    if not file.content_type or not file.content_type.startswith("image/"):
        # 파일 확장자로도 확인
        if file.filename:
            file_extension = file.filename.lower().split('.')[-1]
            allowed_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="업로드된 파일이 이미지 형식이 아닙니다. Content-Type을 확인해주세요.",
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
            # 멀티모달 검증 결과가 있는 경우 - 모든 음식 반환
            food_verifications = mass_estimation["food_verifications"]
            simplified_mass_estimation = {
                "foods": [
                    {
                        "food_name": food_ver.get("food_name"),
                        "estimated_mass_g": food_ver.get("verified_mass_g"),
                        "confidence": food_ver.get("confidence"),
                        "verification_method": food_ver.get("verification_method"),
                        "reasoning": food_ver.get("reasoning", "")
                    }
                    for food_ver in food_verifications
                ],
                "total_mass_g": sum(food_ver.get("verified_mass_g", 0) for food_ver in food_verifications),
                "food_count": len(food_verifications)
            }
        elif "food_estimations" in mass_estimation and mass_estimation["food_estimations"]:
            # 초기 추정 결과가 있는 경우 - 모든 음식 반환
            food_estimations = mass_estimation["food_estimations"]
            simplified_mass_estimation = {
                "foods": [
                    {
                        "food_name": food_est.get("food_name", "알수없음"),
                        "estimated_mass_g": food_est.get("estimated_mass_g"),
                        "confidence": food_est.get("confidence"),
                        "verification_method": "initial_estimation",
                        "reasoning": food_est.get("reasoning", "")
                    }
                    for food_est in food_estimations
                ],
                "total_mass_g": sum(food_est.get("estimated_mass_g", 0) for food_est in food_estimations),
                "food_count": len(food_estimations)
            }
        else:
            # 기존 단일 결과 (하위 호환성)
            simplified_mass_estimation = {
                "foods": [
                    {
                        "food_name": "알수없음",
                        "estimated_mass_g": mass_estimation.get("estimated_mass_g"),
                        "confidence": mass_estimation.get("confidence"),
                        "verification_method": "basic_estimation",
                        "reasoning": mass_estimation.get("reasoning", "")
                    }
                ],
                "total_mass_g": mass_estimation.get("estimated_mass_g", 0),
                "food_count": 1
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

# === 비동기 엔드포인트 추가 ===

@router.post(
    "/estimate_async",
    response_model=TaskCreateResponse,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
    summary="비동기 음식 질량 추정",
    description="이미지 파일을 업로드하면 비동기적으로 질량 추정 작업을 시작하고, 작업 ID를 반환합니다."
)
async def estimate_mass_async(
    file: UploadFile = File(..., description="질량을 추정할 이미지 파일 (JPG, PNG 등)"),
    service: MassEstimationService = Depends(get_mass_estimation_service),
):
    """비동기적으로 질량 추정 작업을 시작합니다."""
    try:
        # 디버깅을 위한 파일 정보 로깅
        if settings.DEBUG_MODE:
            logging.info(f"비동기 파일 업로드 정보:")
            logging.info(f"  - filename: {file.filename}")
            logging.info(f"  - content_type: {file.content_type}")
            logging.info(f"  - size: {file.size if hasattr(file, 'size') else 'unknown'}")
        
        # 파일 타입 검증 (content_type이 None일 수 있으므로 안전하게 처리)
        if not file.content_type or not file.content_type.startswith("image/"):
            # 파일 확장자로도 확인
            if file.filename:
                file_extension = file.filename.lower().split('.')[-1]
                allowed_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
                if file_extension not in allowed_extensions:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}",
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="업로드된 파일이 이미지 형식이 아닙니다. Content-Type을 확인해주세요.",
                )

        task_id = str(uuid.uuid4())
        
        # 작업 초기화
        task_store[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "message": "작업이 시작되었습니다.",
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None
        }
        
        # 파일 내용을 미리 읽어서 저장
        file_contents = await file.read()
        
        # 비동기 작업 시작 (asyncio.create_task 사용)
        asyncio.create_task(process_estimation_task(task_id, file_contents, file.filename, service))
        
        if settings.DEBUG_MODE:
            logging.info(f"비동기 작업 시작: {task_id}")
        
        # 명시적으로 딕셔너리로 반환
        response_data = {
            "task_id": task_id,
            "status": "pending",
            "message": "질량 추정 작업이 시작되었습니다. /api/v1/task/{task_id}로 상태를 확인하세요."
        }
        
        return TaskCreateResponse(**response_data)
        
    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        # 기타 예외는 500 에러로 변환
        logging.error(f"estimate_async 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"서버 내부 오류가 발생했습니다: {str(e)}"
        )

@router.get(
    "/task/{task_id}",
    response_model=TaskStatus,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
    summary="비동기 작업 상태 확인",
    description="작업 ID로 비동기 작업의 진행 상태를 확인합니다."
)
async def get_task_status(task_id: str):
    """비동기 작업의 상태를 확인합니다."""
    if task_id not in task_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="작업을 찾을 수 없습니다."
        )
    
    task = task_store[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        created_at=task["created_at"],
        completed_at=task.get("completed_at"),
        result=task.get("result"),
        error=task.get("error")
    )

# === WebSocket 엔드포인트 추가 ===

@router.websocket("/ws/task/{task_id}")
async def websocket_task_status(websocket: WebSocket, task_id: str):
    """특정 작업의 실시간 상태를 WebSocket으로 전송"""
    await websocket_manager.connect(websocket, task_id)
    
    try:
        # 초기 상태 전송
        if task_id in task_store:
            initial_status = {
                "type": "task_status",
                "task_id": task_id,
                "data": task_store[task_id]
            }
            await websocket.send_text(json.dumps(initial_status, ensure_ascii=False))
        
        # 연결 유지 (클라이언트가 연결을 끊을 때까지 대기)
        while True:
            # 클라이언트로부터 메시지 수신 (필요시)
            data = await websocket.receive_text()
            # 현재는 단방향 통신이므로 메시지 처리 없음
            # 필요시 여기에 클라이언트 명령 처리 로직 추가
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, task_id)
    except Exception as e:
        logging.error(f"WebSocket 오류: {e}")
        websocket_manager.disconnect(websocket, task_id)

# === 기존 process_estimation_task 함수 수정 ===

async def process_estimation_task(task_id: str, file_contents: bytes, filename: str, service: MassEstimationService):
    """백그라운드에서 질량 추정 작업을 처리합니다."""
    try:
        # 작업 상태 업데이트
        task_store[task_id]["status"] = "processing"
        task_store[task_id]["progress"] = 0.1
        task_store[task_id]["message"] = "이미지 파일을 읽는 중..."
        
        # WebSocket으로 상태 업데이트 전송
        await websocket_manager.send_task_update(task_id, {
            "type": "task_update",
            "task_id": task_id,
            "data": task_store[task_id]
        })
        
        if settings.DEBUG_MODE:
            logging.info(f"작업 {task_id}: 이미지 파일 읽기 시작")
        
        # 파일 내용을 numpy 배열로 변환
        nparr = np.frombuffer(file_contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise Exception("이미지 파일을 처리할 수 없습니다.")
        
        task_store[task_id]["progress"] = 0.3
        task_store[task_id]["message"] = "AI 모델로 이미지를 분석하는 중..."
        
        # WebSocket으로 상태 업데이트 전송
        await websocket_manager.send_task_update(task_id, {
            "type": "task_update",
            "task_id": task_id,
            "data": task_store[task_id]
        })
        
        if settings.DEBUG_MODE:
            logging.info(f"작업 {task_id}: AI 파이프라인 실행 시작")
        
        # 파이프라인 실행 (별도 스레드에서)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(thread_pool, run_pipeline_sync, service, image, filename)
        
        if "error" in result:
            raise Exception(result["error"])
        
        task_store[task_id]["progress"] = 0.8
        task_store[task_id]["message"] = "결과를 정리하는 중..."
        
        # WebSocket으로 상태 업데이트 전송
        await websocket_manager.send_task_update(task_id, {
            "type": "task_update",
            "task_id": task_id,
            "data": task_store[task_id]
        })
        
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
            # 멀티모달 검증 결과가 있는 경우 - 모든 음식 반환
            food_verifications = mass_estimation["food_verifications"]
            simplified_mass_estimation = {
                "foods": [
                    {
                        "food_name": food_ver.get("food_name"),
                        "estimated_mass_g": food_ver.get("verified_mass_g"),
                        "confidence": food_ver.get("confidence"),
                        "verification_method": food_ver.get("verification_method"),
                        "reasoning": food_ver.get("reasoning", "")
                    }
                    for food_ver in food_verifications
                ],
                "total_mass_g": sum(food_ver.get("verified_mass_g", 0) for food_ver in food_verifications),
                "food_count": len(food_verifications)
            }
        elif "food_estimations" in mass_estimation and mass_estimation["food_estimations"]:
            # 초기 추정 결과가 있는 경우 - 모든 음식 반환
            food_estimations = mass_estimation["food_estimations"]
            simplified_mass_estimation = {
                "foods": [
                    {
                        "food_name": food_est.get("food_name", "알수없음"),
                        "estimated_mass_g": food_est.get("estimated_mass_g"),
                        "confidence": food_est.get("confidence"),
                        "verification_method": "initial_estimation",
                        "reasoning": food_est.get("reasoning", "")
                    }
                    for food_est in food_estimations
                ],
                "total_mass_g": sum(food_est.get("estimated_mass_g", 0) for food_est in food_estimations),
                "food_count": len(food_estimations)
            }
        else:
            # 기존 단일 결과 (하위 호환성)
            simplified_mass_estimation = {
                "foods": [
                    {
                        "food_name": "알수없음",
                        "estimated_mass_g": mass_estimation.get("estimated_mass_g"),
                        "confidence": mass_estimation.get("confidence"),
                        "verification_method": "basic_estimation",
                        "reasoning": mass_estimation.get("reasoning", "")
                    }
                ],
                "total_mass_g": mass_estimation.get("estimated_mass_g", 0),
                "food_count": 1
            }

        # 최종 응답 생성
        final_result = EstimationResponse(
            filename=filename,
            detected_objects=detected_objects,
            mass_estimation=simplified_mass_estimation,
        )
        
        task_store[task_id]["progress"] = 1.0
        task_store[task_id]["status"] = "completed"
        task_store[task_id]["message"] = "작업이 완료되었습니다."
        task_store[task_id]["completed_at"] = datetime.now().isoformat()
        task_store[task_id]["result"] = final_result
        
        # WebSocket으로 최종 결과 전송 (dict로 변환)
        completion_message = {
            "type": "task_completed",
            "task_id": task_id,
            "data": {
                "status": task_store[task_id]["status"],
                "progress": task_store[task_id]["progress"],
                "message": task_store[task_id]["message"],
                "created_at": task_store[task_id]["created_at"],
                "completed_at": task_store[task_id]["completed_at"],
                "result": final_result.dict()  # <-- 여기서 dict로 변환
            }
        }
        await websocket_manager.send_task_update(task_id, completion_message)
        
        if settings.DEBUG_MODE:
            logging.info(f"작업 {task_id}: 성공적으로 완료")
        
    except Exception as e:
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["error"] = str(e)
        task_store[task_id]["completed_at"] = datetime.now().isoformat()
        
        # WebSocket으로 오류 상태 전송
        await websocket_manager.send_task_update(task_id, {
            "type": "task_failed",
            "task_id": task_id,
            "data": task_store[task_id]
        })
        
        if settings.DEBUG_MODE:
            logging.error(f"작업 {task_id}: 실패 - {e}") 