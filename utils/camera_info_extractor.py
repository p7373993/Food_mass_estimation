"""
카메라 정보 추출기 - EXIF에서 간단한 카메라 정보만 추출
"""

import logging
from typing import Dict, Optional, Tuple
from PIL import Image, ExifTags
import math

class CameraInfoExtractor:
    """
    이미지에서 카메라 정보를 추출하는 클래스 (간소화된 버전)
    """
    
    def __init__(self):
        """카메라 정보 추출기 초기화"""
        pass
    
    def extract_focal_length_only(self, image_path: str) -> Dict:
        """
        이미지에서 초점거리만 간단히 추출 (LLM용)
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            초점거리 정보만 포함한 딕셔너리
        """
        try:
            with Image.open(image_path) as img:
                # EXIF 데이터 추출
                exif_data = img._getexif()
                
                if exif_data is None:
                    return {
                        "has_focal_length": False,
                        "focal_length_mm": 0.0,
                        "focal_length_35mm": 0.0,
                        "camera_type": "unknown"
                    }
                
                # EXIF 태그 매핑
                exif_dict = {}
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = value
                
                # 초점거리 추출
                focal_length = self._extract_focal_length(exif_dict)
                focal_length_35mm = self._extract_focal_length_35mm(exif_dict)
                
                # 카메라 타입 간단 추정
                camera_make = exif_dict.get("Make", "").lower()
                camera_model = exif_dict.get("Model", "").lower()
                camera_type = self._simple_camera_type(camera_make, camera_model)
                
                return {
                    "has_focal_length": focal_length > 0,
                    "focal_length_mm": focal_length,
                    "focal_length_35mm": focal_length_35mm,
                    "camera_type": camera_type
                }
                
        except Exception as e:
            logging.warning(f"초점거리 추출 실패: {e}")
            return {
                "has_focal_length": False,
                "focal_length_mm": 0.0,
                "focal_length_35mm": 0.0,
                "camera_type": "unknown"
            }
    
    def _extract_focal_length(self, exif_dict: Dict) -> float:
        """초점거리 추출"""
        try:
            focal_length = exif_dict.get("FocalLength", 0)
            if isinstance(focal_length, tuple):
                return focal_length[0] / focal_length[1]
            return float(focal_length)
        except:
            return 0.0
    
    def _extract_focal_length_35mm(self, exif_dict: Dict) -> float:
        """35mm 환산 초점거리 추출"""
        try:
            focal_length_35mm = exif_dict.get("FocalLengthIn35mmFilm", 0)
            return float(focal_length_35mm)
        except:
            return 0.0
    
    def _simple_camera_type(self, make: str, model: str) -> str:
        """간단한 카메라 타입 분류"""
        if "iphone" in model or "apple" in make:
            return "smartphone"
        elif "galaxy" in model or "samsung" in make:
            return "smartphone"
        elif "pixel" in model or "google" in make:
            return "smartphone"
        elif any(phone_brand in make or phone_brand in model 
                for phone_brand in ["xiaomi", "huawei", "lg", "sony", "oneplus"]):
            return "smartphone"
        elif any(dslr_brand in make 
                for dslr_brand in ["canon", "nikon", "sony", "fujifilm", "olympus", "panasonic"]):
            return "camera"
        else:
            return "unknown" 