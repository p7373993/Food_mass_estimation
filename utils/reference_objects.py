from typing import Dict, List, Optional, Tuple
import logging
import math
import json
import os
from config.settings import settings

class ReferenceObjectManager:
    """
    기준 물체 정보를 관리하는 클래스 (간소화된 버전)
    """
    
    def __init__(self, db_path: str = "data/reference_objects.json"):
        """
        기준 물체 관리자 초기화
        
        Args:
            db_path: 기준 물체 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.reference_objects = self._load_reference_objects()
        
    def _load_reference_objects(self) -> Dict:
        """기준 물체 데이터베이스 로드"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logging.warning(f"기준 물체 데이터베이스 파일이 없습니다: {self.db_path}")
                return self._create_default_database()
        except Exception as e:
            logging.error(f"기준 물체 데이터베이스 로드 오류: {e}")
            return self._create_default_database()
    
    def _create_default_database(self) -> Dict:
        """기본 기준 물체 데이터베이스 생성"""
        return {
            "earphone_case": {
                "name": "이어폰 케이스",
                "type": "rectangular",
                "width": 5.0,  # cm
                "height": 5.0,  # cm
                "thickness": settings.DEFAULT_REFERENCE_THICKNESS_CM,  # cm
                "area": settings.DEFAULT_REFERENCE_AREA_CM2,  # cm²
                "volume": settings.DEFAULT_REFERENCE_VOLUME_CM3,  # cm³
                "weight": settings.DEFAULT_MASS,  # g
                "description": "사용자의 이어폰 케이스",
                "accuracy": 0.95,
                "common_variations": {}
            }
        }
    
    def get_reference_object(self, object_name: str) -> Optional[Dict]:
        """
        기준 물체 정보 조회
        
        Args:
            object_name: 물체 이름
            
        Returns:
            기준 물체 정보 딕셔너리 또는 None
        """
        return self.reference_objects.get(object_name)
    
    def calculate_scale_factor(self, detected_object: Dict, reference_name: str) -> Optional[float]:
        """
        감지된 물체와 기준 물체를 비교하여 스케일 팩터 계산
        
        Args:
            detected_object: 감지된 물체 정보
            reference_name: 기준 물체 이름
            
        Returns:
            스케일 팩터 (픽셀/cm) 또는 None
        """
        try:
            ref_info = self.get_reference_object(reference_name)
            if not ref_info:
                return None
            
            # 직사각형 물체의 스케일 팩터 계산 (이어폰 케이스 전용)
            if ref_info.get("type") == "rectangular":
                return self._calculate_rectangular_scale(detected_object, ref_info)
            else:
                return self._calculate_general_scale(detected_object, ref_info)
                
        except Exception as e:
            logging.error(f"스케일 팩터 계산 오류: {e}")
            return None
    
    def _calculate_rectangular_scale(self, detected_object: Dict, ref_info: Dict) -> float:
        """직사각형 물체의 스케일 팩터 계산"""
        # 픽셀 면적에서 대략적인 크기 계산
        pixel_area = detected_object.get("pixel_area", 0)
        pixel_size = math.sqrt(pixel_area)
        
        # 실제 면적에서 대략적인 크기 계산
        real_area = ref_info.get("area", 1.0)
        real_size = math.sqrt(real_area)
        
        return pixel_size / real_size if real_size > 0 else 1.0
    
    def _calculate_general_scale(self, detected_object: Dict, ref_info: Dict) -> float:
        """일반적인 물체의 스케일 팩터 계산"""
        # 픽셀 면적에서 대략적인 크기 계산
        pixel_area = detected_object.get("pixel_area", 0)
        pixel_size = math.sqrt(pixel_area)
        
        # 실제 면적에서 대략적인 크기 계산
        real_area = ref_info.get("area", 1.0)
        real_size = math.sqrt(real_area)
        
        return pixel_size / real_size if real_size > 0 else 1.0
    
    def get_best_reference_objects(self, detected_objects: List[Dict]) -> List[Tuple[str, float]]:
        """
        감지된 물체들 중 가장 좋은 기준 물체들 선택
        
        Args:
            detected_objects: 감지된 물체들 리스트
            
        Returns:
            (물체명, 정확도) 튜플들의 리스트
        """
        best_references = []
        
        for obj in detected_objects:
            obj_name = obj.get("class_name", "")
            if obj_name in self.reference_objects:
                ref_info = self.reference_objects[obj_name]
                accuracy = ref_info.get("accuracy", settings.DEFAULT_CONFIDENCE)
                confidence = obj.get("confidence", settings.DEFAULT_CONFIDENCE)
                
                # 전체 신뢰도 = 기준 물체 정확도 × 감지 신뢰도
                total_confidence = accuracy * confidence
                best_references.append((obj_name, total_confidence))
        
        # 신뢰도순으로 정렬
        best_references.sort(key=lambda x: x[1], reverse=True)
        
        return best_references
    
    def suggest_reference_objects(self, detected_classes: List[str]) -> List[str]:
        """
        감지된 클래스들을 기반으로 추천 기준 물체 제안
        
        Args:
            detected_classes: 감지된 클래스들 리스트
            
        Returns:
            추천 기준 물체들 리스트
        """
        suggestions = []
        
        # 기준 물체가 감지되지 않은 경우 추천
        reference_names = set(self.reference_objects.keys())
        detected_references = set(detected_classes) & reference_names
        
        if not detected_references:
            # 정확도가 높은 순으로 추천
            sorted_refs = sorted(
                self.reference_objects.items(),
                key=lambda x: x[1].get("accuracy", settings.DEFAULT_CONFIDENCE),
                reverse=True
            )
            suggestions = [name for name, _ in sorted_refs[:3]]
        
        return suggestions
    
    def get_statistics(self) -> Dict:
        """기준 물체 데이터베이스 통계 반환"""
        total_objects = len(self.reference_objects)
        types_count = {}
        avg_accuracy = 0
        
        for obj_info in self.reference_objects.values():
            obj_type = obj_info.get("type", "unknown")
            types_count[obj_type] = types_count.get(obj_type, 0) + 1
            avg_accuracy += obj_info.get("accuracy", 0)
        
        avg_accuracy = avg_accuracy / total_objects if total_objects > 0 else 0
        
        return {
            "total_objects": total_objects,
            "types_distribution": types_count,
            "average_accuracy": avg_accuracy,
            "available_types": list(types_count.keys())
        } 