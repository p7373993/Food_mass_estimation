import logging
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageOps
from ultralytics import YOLO
from typing import Dict, Tuple, List

# 프로젝트 루트를 기준으로 config.settings를 임포트
from config.settings import settings
from utils.base_model import BaseModel

class YOLOSegmentationModel(BaseModel):
    """
    YOLOv8 분할 모델을 관리하는 래퍼 클래스.
    - BaseModel을 상속받아 싱글톤 패턴과 공통 로직 사용
    - 중앙 설정 파일(settings)을 사용합니다.
    - 기존의 결과 파싱 및 시각화 기능을 유지합니다.
    """
    
    def __init__(self):
        self.class_mapping = {
            0: "food",
            1: "earphone_case" 
        }
        super().__init__()
    
    def get_model_name(self) -> str:
        return "YOLO 세그멘테이션 모델"
    
    def _initialize_model(self) -> None:
        """YOLO 모델 초기화"""
        try:
            self._model = YOLO(settings.YOLO_MODEL_PATH)
            self._model.to(self.device)
            self._log_success(f"로딩 성공: {settings.YOLO_MODEL_PATH} -> {self.device}")
        except Exception as e:
            self._log_error("로딩 실패", e)
            self._model = None
    
    def segment_image(self, image: np.ndarray) -> Dict:
        """
        이미지에서 객체 세그멘테이션 수행
        
        Args:
            image: 입력 이미지 (numpy array)
            
        Returns:
            세그멘테이션 결과 딕셔너리
        """
        if self._model is None:
            self._load_model()

        try:
            # 모델 예측 (confidence threshold는 _parse_results에서 처리)
            verbose = not settings.DEBUG_MODE
            results = self._model(image, verbose=verbose)
            
            # 결과 파싱
            segmentation_results = self._parse_results(results[0], results[0].masks.data.shape[1:] if results[0].masks is not None else image.shape[:2])
            
            return {
                "image_shape": image.shape,
                "food_objects": segmentation_results["food_objects"],
                "reference_objects": segmentation_results["reference_objects"],
                "all_objects": segmentation_results["all_objects"]
            }
            
        except Exception as e:
            logging.error(f"세그멘테이션 실행 중 오류: {e}")
            raise

    def _parse_results(self, result, image_shape: Tuple) -> Dict:
        """
        YOLO 결과를 파싱하여 구조화된 데이터로 변환 (사용자 제공 안정 버전 기반)
        
        Args:
            result: YOLO 모델의 예측 결과
            image_shape: 원본 이미지 크기
            
        Returns:
            파싱된 결과 딕셔너리
        """
        food_objects = []
        reference_objects = []
        all_objects = []
        
        # 결과가 없는 경우 즉시 반환
        if result.masks is None or result.boxes is None:
            return {
                "food_objects": food_objects,
                "reference_objects": reference_objects,
                "all_objects": all_objects
            }

        # .data 속성을 사용하여 raw tensor 추출 후 numpy로 변환
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()
        
        confidence_threshold = settings.CONFIDENCE_THRESHOLD

        for i, box in enumerate(boxes):
            confidence = box[4]
            
            if confidence < confidence_threshold:
                continue
            
            class_id = int(box[5])
            
            # 마스크 처리 및 리사이즈
            mask = masks[i]
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # 디버그 정보 추가
            if settings.DEBUG_MODE:
                print(f"마스크 {i}: 원본 크기 {mask.shape}, 이진화 후 크기 {mask_binary.shape}")
                print(f"마스크 {i}: 이진화 전 최소값 {mask.min():.3f}, 최대값 {mask.max():.3f}")
                print(f"마스크 {i}: 이진화 후 1의 개수 {np.sum(mask_binary)}")
            
            # 마스크가 이미 올바른 크기인지 확인 (리사이즈가 필요한 경우에만 수행)
            # YOLO 모델이 이미 적절한 크기로 처리했으므로 불필요한 리사이즈 방지
            if mask_binary.shape != image_shape[:2] and image_shape[0] > 0 and image_shape[1] > 0:
                original_size = mask_binary.shape
                mask_binary = cv2.resize(mask_binary, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
                if settings.DEBUG_MODE:
                    print(f"마스크 {i}: 리사이즈 {original_size} -> {mask_binary.shape}")
                    print(f"마스크 {i}: 리사이즈 후 1의 개수 {np.sum(mask_binary)}")
            elif settings.DEBUG_MODE:
                print(f"마스크 {i}: 리사이즈 불필요 (이미 올바른 크기)")

            # 바운딩 박스 좌표
            x1, y1, x2, y2 = box[:4].astype(int)
            
            # 픽셀 면적 계산
            pixel_area = np.sum(mask_binary)
            
            if settings.DEBUG_MODE:
                print(f"객체 {i} ({self._get_class_name(class_id)}): 픽셀 면적 {pixel_area:,}")
            
            # 객체 정보 생성
            obj_info = {
                "class_id": class_id,
                "class_name": self._get_class_name(class_id),
                "confidence": float(confidence),
                "bbox": [x1, y1, x2, y2],
                "pixel_area": int(pixel_area),
                "mask": mask_binary,
                "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                "position": {
                    "x": (x1 + x2) // 2,
                    "y": (y1 + y2) // 2,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            }
            
            all_objects.append(obj_info)
            
            # 음식 vs 기준 물체 분류
            class_name = self._get_class_name(class_id)
            if class_name == "food":
                food_objects.append(obj_info)
            elif class_name == "earphone_case":
                reference_objects.append(obj_info)
    
        return {
            "food_objects": food_objects,
            "reference_objects": reference_objects,
            "all_objects": all_objects
        }

    def _get_class_name(self, class_id: int) -> str:
        """클래스 ID를 클래스 이름으로 변환"""
        return self.class_mapping.get(class_id, f"unknown_{class_id}")

# 싱글톤 인스턴스 생성
yolo_model = YOLOSegmentationModel()

def load_image(image_path: str | Path) -> np.ndarray | None:
    """EXIF 정보를 처리하며 이미지를 로드하고 BGR 형식으로 변환"""
    try:
        pil_image = Image.open(image_path)
        pil_image = ImageOps.exif_transpose(pil_image)
        # RGB -> BGR로 변환
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"이미지 로드 실패: {image_path}, 오류: {e}")
        return None 