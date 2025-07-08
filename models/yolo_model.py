import torch
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from PIL import Image, ImageOps
from pipeline.config import Config

class YOLOSegmentationModel:
    """
    YOLO 세그멘테이션 모델을 사용하여 음식과 기준 물체를 분할하는 클래스
    """
    
    def __init__(self, model_path: str = "yolo_food.pt"):
        """
        YOLO 모델 초기화
        
        Args:
            model_path: 파인튜닝된 YOLO 모델 경로
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 클래스 매핑 (단순화)
        # 0: 음식, 1: 기준물체(이어폰 케이스)
        self.class_mapping = {
            0: "food",
            1: "earphone_case"
        }
        
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            # 간단 디버그 모드에서 verbose 비활성화
            if hasattr(Config, 'SIMPLE_DEBUG') and Config.SIMPLE_DEBUG:
                self.model.verbose = False
            logging.info(f"YOLO 모델이 성공적으로 로드되었습니다: {self.model_path}")
        except Exception as e:
            logging.error(f"모델 로드 실패: {e}")
            raise
    
    def segment_image(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        이미지에서 객체 세그멘테이션 수행
        
        Args:
            image_path: 입력 이미지 경로
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            세그멘테이션 결과 딕셔너리
        """
        try:
            # 이미지 읽기 - EXIF 정보 처리
            pil_image = Image.open(image_path)
            pil_image = ImageOps.exif_transpose(pil_image)  # EXIF 회전 정보 처리
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 모델 예측
            verbose = not (hasattr(Config, 'SIMPLE_DEBUG') and Config.SIMPLE_DEBUG)
            results = self.model(image, verbose=verbose)
            
            # 결과 파싱
            segmentation_results = self._parse_results(results[0], image.shape, confidence_threshold)
            
            return {
                "image_shape": image.shape,
                "food_objects": segmentation_results["food_objects"],
                "reference_objects": segmentation_results["reference_objects"],
                "all_objects": segmentation_results["all_objects"]
            }
            
        except Exception as e:
            logging.error(f"세그멘테이션 실행 중 오류: {e}")
            raise
    
    def _parse_results(self, result, image_shape: Tuple, confidence_threshold: float) -> Dict:
        """
        YOLO 결과를 파싱하여 구조화된 데이터로 변환
        
        Args:
            result: YOLO 모델의 예측 결과
            image_shape: 원본 이미지 크기
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            파싱된 결과 딕셔너리
        """
        food_objects = []
        reference_objects = []
        all_objects = []
        
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                confidence = box[4]
                class_id = int(box[5])
                
                if confidence < confidence_threshold:
                    continue
                
                # 마스크 처리
                mask_binary = (mask > 0.5).astype(np.uint8)
                
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = box[:4].astype(int)
                
                # 픽셀 면적 계산
                pixel_area = np.sum(mask_binary)
                
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
                if class_id == 0:  # 음식
                    food_objects.append(obj_info)
                elif class_id == 1:  # 이어폰 케이스
                    reference_objects.append(obj_info)
        
        return {
            "food_objects": food_objects,
            "reference_objects": reference_objects,
            "all_objects": all_objects
        }
    
    def _get_class_name(self, class_id: int) -> str:
        """클래스 ID를 클래스 이름으로 변환"""
        return self.class_mapping.get(class_id, f"unknown_{class_id}")
    
    def visualize_segmentation(self, image_path: str, save_path: str = None, show_masks: bool = True, 
                              show_boxes: bool = True, alpha: float = 0.5) -> np.ndarray:
        """
        세그멘테이션 결과를 시각화
        
        Args:
            image_path: 입력 이미지 경로
            save_path: 저장할 경로 (선택사항)
            show_masks: 마스크 표시 여부
            show_boxes: 바운딩 박스 표시 여부
            alpha: 마스크 투명도 (0.0 ~ 1.0)
            
        Returns:
            시각화된 이미지
        """
        try:
            # 이미지 읽기 - EXIF 정보 처리
            pil_image = Image.open(image_path)
            pil_image = ImageOps.exif_transpose(pil_image)  # EXIF 회전 정보 처리
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            segmentation_results = self.segment_image(image_path)
            
            # 시각화 이미지 생성
            vis_image = image.copy()
            overlay = image.copy()
            
            # 음식 객체 시각화 (빨간색 계열)
            food_color = (0, 0, 255)  # 빨간색
            for i, obj in enumerate(segmentation_results["food_objects"]):
                if show_masks:
                    mask = obj["mask"]
                    # 마스크 크기를 원본 이미지 크기에 맞춤
                    if mask.shape != image.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), 
                                        (image.shape[1], image.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                    
                    # 반투명 마스크 적용
                    overlay[mask > 0] = food_color
                
                if show_boxes:
                    # 바운딩 박스 그리기
                    x1, y1, x2, y2 = obj["bbox"]
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), food_color, 3)
                    
                    # 라벨 배경
                    label = f"Food #{i+1}: {obj['confidence']:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(vis_image, (x1, y1-label_size[1]-10), 
                                (x1+label_size[0], y1), food_color, -1)
                    
                    # 라벨 텍스트
                    cv2.putText(vis_image, label, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 기준 물체 시각화 (파란색 계열)
            ref_color = (255, 100, 0)  # 주황색-파란색
            for i, obj in enumerate(segmentation_results["reference_objects"]):
                if show_masks:
                    mask = obj["mask"]
                    # 마스크 크기를 원본 이미지 크기에 맞춤
                    if mask.shape != image.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), 
                                        (image.shape[1], image.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                    
                    # 반투명 마스크 적용
                    overlay[mask > 0] = ref_color
                
                if show_boxes:
                    # 바운딩 박스 그리기
                    x1, y1, x2, y2 = obj["bbox"]
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), ref_color, 3)
                    
                    # 라벨 배경
                    label = f"Reference #{i+1}: {obj['confidence']:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(vis_image, (x1, y1-label_size[1]-10), 
                                (x1+label_size[0], y1), ref_color, -1)
                    
                    # 라벨 텍스트
                    cv2.putText(vis_image, label, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 마스크와 원본 이미지 블렌딩
            if show_masks:
                vis_image = cv2.addWeighted(vis_image, 1-alpha, overlay, alpha, 0)
            
            # 범례 추가
            legend_y = 30
            cv2.putText(vis_image, "Segmentation Results:", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, f"Food: {len(segmentation_results['food_objects'])} detected", 
                       (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, food_color, 2)
            cv2.putText(vis_image, f"Reference: {len(segmentation_results['reference_objects'])} detected", 
                       (10, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ref_color, 2)
            
            # 이미지 저장
            if save_path:
                # 디렉토리가 없으면 생성
                import os
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                cv2.imwrite(save_path, vis_image)
                logging.info(f"세그멘테이션 시각화 저장: {save_path}")
            
            return vis_image
            
        except Exception as e:
            logging.error(f"시각화 중 오류: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "class_mapping": self.class_mapping
        } 