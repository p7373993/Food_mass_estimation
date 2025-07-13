import cv2
import torch
import numpy as np
import logging
from typing import Dict
from PIL import Image, ImageOps

from config.settings import settings
from utils.base_model import BaseModel

class MiDaSDepthModel(BaseModel):
    """
    MiDaS 깊이 추정 모델을 관리하는 래퍼 클래스.
    - BaseModel을 상속받아 싱글톤 패턴과 공통 로직 사용
    - 중앙 설정 파일(settings)을 사용합니다.
    - 기존의 객체별 깊이 분석 및 상대 크기 추정 기능을 유지합니다.
    """
    
    def __init__(self):
        self._transform = None
        super().__init__()
    
    def get_model_name(self) -> str:
        return "MiDaS 깊이 추정 모델"
    
    def _initialize_model(self) -> None:
        """MiDaS 모델 초기화"""
        model_type = settings.MIDAS_MODEL_TYPE
        
        try:
            self._model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            self._model.to(self.device)
            self._model.eval()

            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._transform = transforms.dpt_transform if "dpt" in model_type.lower() else transforms.small_transform
            
            self._log_success(f"로딩 성공: {model_type} -> {self.device}")
        except Exception as e:
            self._log_error("로딩 실패", e)
            self._model = None

    def estimate_depth(self, image: np.ndarray) -> np.ndarray | None:
        """
        주어진 이미지에 대해 깊이 맵을 추정합니다.
        """
        if self._model is None or self._transform is None:
            logging.error("MiDaS 모델이 로드되지 않아 깊이 추정을 수행할 수 없습니다.")
            return None

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with torch.no_grad():
                transformed_image = self._transform(rgb_image).to(self.device)
                prediction = self._model(transformed_image)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            return prediction.cpu().numpy()

        except Exception as e:
            logging.error(f"MiDaS 깊이 추정 중 오류 발생: {e}")
            return None

    def get_object_depth_info(self, depth_map: np.ndarray, mask: np.ndarray) -> Dict:
        """
        특정 객체(마스크)의 깊이 정보 추출
        """
        try:
            object_depths = depth_map[mask > 0]
            if object_depths.size == 0:
                return {"mean_depth": 0.0, "depth_variation": 0.0}

            return {
                "mean_depth": float(np.mean(object_depths)),
                "depth_variation": float(np.ptp(object_depths)) # peak-to-peak (max-min)
            }
        except Exception as e:
            logging.error(f"객체 깊이 정보 추출 중 오류: {e}")
            return {"mean_depth": 0.0, "depth_variation": 0.0}

    def estimate_relative_size(self, depth_map: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> Dict:
        """
        두 객체 간의 상대적 크기 추정
        """
        try:
            obj1_depth_info = self.get_object_depth_info(depth_map, mask1)
            obj2_depth_info = self.get_object_depth_info(depth_map, mask2)

            area1 = np.sum(mask1)
            area2 = np.sum(mask2)

            if area2 == 0 or obj2_depth_info["mean_depth"] == 0:
                return {"corrected_area_ratio": 1.0}

            depth_ratio = obj1_depth_info["mean_depth"] / obj2_depth_info["mean_depth"]
            
            # (Area1 / Depth1^2) / (Area2 / Depth2^2) = (Area1/Area2) * (Depth2/Depth1)^2
            # 논문/이론에 따르면 깊이의 제곱에 반비례하지만, 실제 테스트에서는 깊이에 비례하는 것이 더 나은 결과를 줄 수 있음.
            # 여기서는 이론에 따라 제곱으로 계산.
            corrected_area_ratio = (area1 / area2) * (1 / (depth_ratio ** 2))

            return {"corrected_area_ratio": corrected_area_ratio}

        except Exception as e:
            logging.error(f"상대적 크기 추정 중 오류: {e}")
            return {"corrected_area_ratio": 1.0}


# 싱글톤 인스턴스 생성
midas_model = MiDaSDepthModel() 