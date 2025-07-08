import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
import math
from pipeline.config import Config
from .camera_info_extractor import CameraInfoExtractor
from .reference_objects import ReferenceObjectManager

class FeatureExtractor:
    """
    세그멘테이션과 깊이 정보에서 특징을 추출하는 클래스 (간소화된 버전)
    
    이 클래스는 YOLO 세그멘테이션과 MiDaS 깊이 추정 결과를 받아
    LLM에 전달할 핵심 특징들만 추출합니다.
    """
    
    def __init__(self):
        """특징 추출기 초기화"""
        self.reference_manager = ReferenceObjectManager()
        self.camera_info_extractor = CameraInfoExtractor()
    
    def extract_features(self, segmentation_results: Dict, depth_results: Dict, image_path: str = None) -> Dict:
        """
        간소화된 특징 추출 - LLM에게 필요한 핵심 정보만 추출
        
        Args:
            segmentation_results: YOLO 세그멘테이션 결과
            depth_results: MiDaS 깊이 추정 결과
            image_path: 이미지 경로 (선택사항)
            
        Returns:
            간소화된 특징 딕셔너리
        """
        try:
            depth_map = depth_results["depth_map"]
            
            # 1. 음식 객체 기본 특징 추출
            food_features = self._extract_basic_food_features(
                segmentation_results["food_objects"], 
                depth_map
            )
            
            # 2. 기준 물체 기본 특징 추출
            reference_features = self._extract_basic_reference_features(
                segmentation_results["reference_objects"], 
                depth_map
            )
            
            # 3. 깊이 스케일 계산 (기준 물체 기반)
            depth_scale_info = self._calculate_depth_scale(reference_features, depth_map)
            
            # 4. 기본 상대 크기 정보만 계산
            relative_size_info = self._calculate_basic_relative_sizes(
                food_features, 
                reference_features
            )
            
            # 5. 초점거리 정보만 간단히 추출 (이미지 경로가 있는 경우)
            focal_length_info = None
            if image_path:
                focal_length_info = self.camera_info_extractor.extract_focal_length_only(image_path)
            
            # 6. 예외 처리 및 대안적 계산 정보 추가
            fallback_info = self._calculate_fallback_info(
                food_features, reference_features, depth_scale_info, focal_length_info
            )
            
            # 핵심 정보만 반환
            return {
                "food_objects": food_features,
                "reference_objects": reference_features,
                "depth_scale_info": depth_scale_info,
                "relative_size_info": relative_size_info,
                "focal_length_info": focal_length_info,
                "fallback_info": fallback_info,
                "image_shape": segmentation_results["image_shape"]
            }
            
        except Exception as e:
            logging.error(f"특징 추출 중 오류: {e}")
            # 기본값 반환
            return {
                "food_objects": [],
                "reference_objects": [],
                "depth_scale_info": {"has_scale": False, "method": "none"},
                "relative_size_info": [],
                "focal_length_info": None,
                "fallback_info": {"method": "error", "confidence": 0.1},
                "image_shape": (640, 480)
            }
    
    def _extract_basic_food_features(self, food_objects: List[Dict], depth_map: np.ndarray) -> List[Dict]:
        """음식 객체의 기본 특징만 추출"""
        features = []
        
        for food_obj in food_objects:
            mask = food_obj["mask"]
            
            # 기본 깊이 정보만 추출
            depth_info = self._get_basic_depth_info(depth_map, mask)
            
            # 간단한 부피 추정
            volume_estimate = self._estimate_basic_volume(mask, depth_info)
            
            feature = {
                "class_id": food_obj["class_id"],
                "class_name": food_obj["class_name"],
                "confidence": food_obj["confidence"],
                "bbox": food_obj["bbox"],
                "pixel_area": food_obj["pixel_area"],
                "depth_info": depth_info,
                "volume_estimate": volume_estimate,
                "mask": mask  # 마스크 정보 보존
            }
            
            features.append(feature)
        
        return features
    
    def _enhance_food_features_with_reference(self, food_features: List[Dict], reference_features: List[Dict]) -> List[Dict]:
        """기준 물체를 활용하여 음식 특징을 개선"""
        if not reference_features:
            return food_features
        
        enhanced_features = []
        
        for food in food_features:
            # 기존 특징 복사
            enhanced_food = food.copy()
            
            # 기준 물체를 활용한 정확한 부피 계산
            mask = food.get("mask")  # 원본 마스크가 필요한 경우
            depth_info = food.get("depth_info", {})
            
            if mask is not None:
                # 기준 물체 기반 부피 계산
                reference_volume = self._calculate_volume_with_reference_scaling(
                    mask, depth_info, reference_features
                )
                
                # 실제 부피 정보 추가
                enhanced_food["real_volume_info"] = reference_volume
                
                # 기존 부피 추정보다 정확도가 높으면 교체
                if reference_volume.get("confidence", 0) > food["volume_estimate"].get("confidence", 0):
                    enhanced_food["volume_estimate"] = reference_volume
            
            enhanced_features.append(enhanced_food)
        
        return enhanced_features
    
    def _extract_basic_reference_features(self, reference_objects: List[Dict], depth_map: np.ndarray) -> List[Dict]:
        """기준 물체의 기본 특징만 추출"""
        features = []
        
        for ref_obj in reference_objects:
            mask = ref_obj["mask"]
            class_name = ref_obj["class_name"]
            
            # 기본 깊이 정보만 추출
            depth_info = self._get_basic_depth_info(depth_map, mask)
            
            # 기준 물체 정보 가져오기
            real_size = self.reference_manager.get_reference_object(class_name)
            if not real_size:
                # 기본값 대신 경고 로그
                logging.warning(f"기준 물체 '{class_name}' 정보를 찾을 수 없습니다.")
                real_size = {"width": 5.0, "height": 5.0, "thickness": 2.5, "area": 25.0, "volume": 62.5}
            
            feature = {
                "class_id": ref_obj["class_id"],
                "class_name": class_name,
                "confidence": ref_obj["confidence"],
                "bbox": ref_obj["bbox"],
                "pixel_area": ref_obj["pixel_area"],
                "depth_info": depth_info,
                "real_size": real_size,
                "mask": mask  # 마스크 정보 보존
            }
            
            features.append(feature)
        
        return features
    
    def _get_basic_depth_info(self, depth_map: np.ndarray, mask: np.ndarray) -> Dict:
        """기본 깊이 정보만 추출"""
        try:
            # 마스크 크기 조정
            if mask.shape[:2] != depth_map.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), 
                                (depth_map.shape[1], depth_map.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # 마스크 영역의 깊이 값만 추출
            object_depths = depth_map[mask > 0]
            
            if len(object_depths) == 0:
                return {
                    "mean_depth": 0.0,
                    "depth_variation": 0.0
                }
            
            return {
                "mean_depth": float(object_depths.mean()),
                "depth_variation": float(object_depths.max() - object_depths.min())
            }
            
        except Exception as e:
            logging.error(f"깊이 정보 추출 오류: {e}")
            return {
                "mean_depth": 0.0,
                "depth_variation": 0.0
            }
    
    def _estimate_basic_volume(self, mask: np.ndarray, depth_info: Dict) -> Dict:
        """개선된 부피 추정 로직"""
        try:
            # 픽셀 면적
            pixel_area = np.sum(mask > 0)
            
            # 깊이 정보 활용
            mean_depth = depth_info.get("mean_depth", 0.0)
            depth_variation = depth_info.get("depth_variation", 1.0)
            
            # 개선된 부피 계산
            # 1. 단순 부피 (픽셀 면적 × 깊이 변화)
            simple_volume = pixel_area * depth_variation
            
            # 2. 형태 보정 부피 (음식의 일반적인 형태 고려)
            # 대부분의 음식은 완전한 원기둥이 아니라 불규칙한 형태
            shape_factor = 0.6  # 경험적 보정 계수
            shape_corrected_volume = simple_volume * shape_factor
            
            # 3. 실제 크기 추정 (대략적인 스케일링)
            # 일반적인 음식 크기를 고려한 스케일링
            # 평균적으로 1픽셀 = 0.5mm 정도로 가정
            pixel_to_mm = 0.5
            pixel_to_cm = pixel_to_mm / 10.0  # 0.05cm
            
            # 실제 부피 계산 (cm³)
            estimated_volume_cm3 = shape_corrected_volume * (pixel_to_cm ** 3)
            
            # 4. 합리성 검증 및 보정
            # 너무 작거나 큰 값은 보정
            if estimated_volume_cm3 < 5.0:  # 5cm³ 이하
                estimated_volume_cm3 = 20.0  # 최소 20cm³
                confidence = 0.3
            elif estimated_volume_cm3 > 1000.0:  # 1000cm³ 이상
                estimated_volume_cm3 = 500.0  # 최대 500cm³
                confidence = 0.4
            else:
                confidence = 0.7
            
            return {
                "pixel_volume": shape_corrected_volume,
                "volume_cm3": estimated_volume_cm3,
                "confidence": confidence,
                "calculation_method": "improved_shape_corrected",
                "shape_factor": shape_factor,
                "pixel_to_cm": pixel_to_cm
            }
            
        except Exception as e:
            logging.error(f"부피 추정 오류: {e}")
            return {
                "pixel_volume": 1000.0,
                "volume_cm3": 50.0,
                "confidence": 0.3,
                "calculation_method": "fallback"
            }
    
    def _calculate_basic_relative_sizes(self, food_features: List[Dict], reference_features: List[Dict]) -> List[Dict]:
        """개선된 상대 크기 정보 계산"""
        if not food_features or not reference_features:
            return []
        
        relative_info = []
        
        for food in food_features:
            for ref in reference_features:
                # 기본 면적 비율
                area_ratio = food["pixel_area"] / ref["pixel_area"] if ref["pixel_area"] > 0 else 1.0
                
                # 실제 크기 비율 계산 (기준 물체가 있는 경우)
                real_size_ratio = None
                if food.get("real_volume_info") and ref.get("real_size"):
                    food_volume = food["real_volume_info"].get("volume_cm3", 0)
                    ref_volume = ref["real_size"].get("volume", 62.5)  # 이어폰 케이스 62.5cm³
                    
                    if ref_volume > 0:
                        real_size_ratio = food_volume / ref_volume
                
                # 깊이 비율 (높이 비교)
                food_depth_var = food.get("depth_info", {}).get("depth_variation", 1.0)
                ref_depth_var = ref.get("depth_info", {}).get("depth_variation", 1.0)
                depth_ratio = food_depth_var / ref_depth_var if ref_depth_var > 0 else 1.0
                
                relative_data = {
                    "food_class": food["class_name"],
                    "reference_class": ref["class_name"],
                    "pixel_area_ratio": area_ratio,
                    "depth_ratio": depth_ratio
                }
                
                # 실제 크기 비율이 계산된 경우 추가
                if real_size_ratio is not None:
                    relative_data["real_size_ratio"] = real_size_ratio
                    relative_data["calculation_method"] = "reference_scaled"
                    relative_data["confidence"] = 0.8
                else:
                    relative_data["calculation_method"] = "pixel_only"
                    relative_data["confidence"] = 0.5
                
                relative_info.append(relative_data)
        
        return relative_info
    
    def _calculate_volume_with_reference_scaling(self, mask: np.ndarray, depth_info: Dict, 
                                              reference_features: List[Dict]) -> Dict:
        """개선된 기준 물체 기반 부피 계산"""
        try:
            if not reference_features:
                return self._estimate_basic_volume(mask, depth_info)
            
            # 가장 신뢰도가 높은 기준 물체 선택
            best_ref = max(reference_features, key=lambda x: x.get("confidence", 0))
            ref_real_size = best_ref.get("real_size", {})
            ref_pixel_area = best_ref.get("pixel_area", 1)
            
            # 스케일 계산 개선
            ref_real_area = ref_real_size.get("area", 25.0)  # 5×5=25cm²
            if ref_pixel_area > 0 and ref_real_area > 0:
                pixel_per_cm2 = ref_pixel_area / ref_real_area
                pixel_per_cm = math.sqrt(pixel_per_cm2)
            else:
                logging.warning("기준 물체 정보가 불충분하여 기본 계산 방식 사용")
                return self._estimate_basic_volume(mask, depth_info)
            
            # 음식 객체의 실제 면적 계산
            food_pixel_area = np.sum(mask > 0)
            food_real_area_cm2 = food_pixel_area / pixel_per_cm2
            
            # 높이 계산 개선
            depth_variation = depth_info.get("depth_variation", 1.0)
            ref_depth_variation = best_ref.get("depth_info", {}).get("depth_variation", 1.0)
            
            # 기준 물체의 실제 높이 (이어폰 케이스: 2.5cm)
            ref_real_height = ref_real_size.get("thickness", 2.5)
            
            if ref_depth_variation > 0:
                # 깊이 변화 비율로 실제 높이 계산
                height_scale = ref_real_height / ref_depth_variation
                estimated_height_cm = depth_variation * height_scale
            else:
                # 픽셀 스케일로 직접 계산
                estimated_height_cm = depth_variation / pixel_per_cm
            
            # 음식 종류별 형태 보정
            shape_factor = self._get_shape_factor_by_food_type(mask)
            
            # 부피 계산
            estimated_volume_cm3 = food_real_area_cm2 * estimated_height_cm * shape_factor
            
            # 합리성 검증 및 신뢰도 계산
            confidence = self._calculate_volume_confidence(
                estimated_volume_cm3, food_real_area_cm2, estimated_height_cm, best_ref
            )
            
            # 극단적인 값 보정
            if estimated_volume_cm3 < 5.0:
                estimated_volume_cm3 = 15.0
                confidence *= 0.6
            elif estimated_volume_cm3 > 1000.0:
                estimated_volume_cm3 = 600.0
                confidence *= 0.7
            
            return {
                "volume_cm3": estimated_volume_cm3,
                "real_area_cm2": food_real_area_cm2,
                "estimated_height_cm": estimated_height_cm,
                "confidence": confidence,
                "calculation_method": "enhanced_reference_scaled",
                "pixel_per_cm": pixel_per_cm,
                "shape_factor": shape_factor,
                "reference_object": best_ref["class_name"],
                "reference_confidence": best_ref.get("confidence", 0)
            }
            
        except Exception as e:
            logging.error(f"개선된 기준 물체 기반 부피 계산 오류: {e}")
            return self._estimate_basic_volume(mask, depth_info)
    
    def _get_shape_factor_by_food_type(self, mask: np.ndarray) -> float:
        """음식 종류별 형태 보정 계수"""
        try:
            # 마스크의 형태 분석
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.6  # 기본값
            
            # 가장 큰 컨투어 선택
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 원형도 계산
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                
                # 원형도에 따른 형태 보정
                if circularity > 0.7:  # 원형에 가까움
                    return 0.65  # 그릇, 둥근 음식
                elif circularity > 0.4:  # 타원형
                    return 0.60  # 일반적인 음식
                else:  # 불규칙한 형태
                    return 0.55  # 복잡한 형태의 음식
            
            return 0.6  # 기본값
            
        except Exception as e:
            logging.error(f"형태 보정 계수 계산 오류: {e}")
            return 0.6
    
    def _calculate_volume_confidence(self, volume_cm3: float, area_cm2: float, 
                                   height_cm: float, ref_obj: Dict) -> float:
        """부피 계산 신뢰도 계산"""
        try:
            base_confidence = 0.7
            
            # 기준 물체의 신뢰도 반영
            ref_confidence = ref_obj.get("confidence", 0.5)
            confidence = base_confidence * (0.5 + ref_confidence * 0.5)
            
            # 계산된 값의 합리성 검증
            if 10 <= volume_cm3 <= 500:  # 합리적인 범위
                confidence *= 1.0
            elif 5 <= volume_cm3 <= 800:  # 허용 가능한 범위
                confidence *= 0.8
            else:  # 극단적인 값
                confidence *= 0.5
            
            # 높이의 합리성 검증
            if 0.5 <= height_cm <= 15:  # 합리적인 높이
                confidence *= 1.0
            elif 0.2 <= height_cm <= 25:  # 허용 가능한 높이
                confidence *= 0.9
            else:  # 극단적인 높이
                confidence *= 0.6
            
            return min(confidence, 0.95)  # 최대 95%로 제한
            
        except Exception as e:
            logging.error(f"신뢰도 계산 오류: {e}")
            return 0.5 

    def _calculate_depth_scale(self, reference_features: List[Dict], depth_map: np.ndarray) -> Dict:
        """
        기준 물체를 이용한 깊이 스케일 계산
        
        Args:
            reference_features: 기준 물체 특징들
            depth_map: 깊이 맵
            
        Returns:
            깊이 스케일 정보 딕셔너리
        """
        try:
            if not reference_features:
                return {
                    "has_scale": False,
                    "method": "no_reference_objects",
                    "depth_scale_cm_per_unit": 0.0,
                    "confidence": 0.0
                }
            
            individual_scales = []
            
            for ref_obj in reference_features:
                # 기준 물체 실제 크기 정보
                real_size = ref_obj.get("real_size", {})
                class_name = ref_obj.get("class_name", "")
                
                # 깊이 정보
                depth_info = ref_obj.get("depth_info", {})
                depth_variation = depth_info.get("depth_variation", 0.0)
                
                # 실제 두께 정보 (이어폰 케이스 등)
                real_thickness = real_size.get("thickness", 0.0)
                
                if depth_variation > 0 and real_thickness > 0:
                    # 깊이 변화량 대비 실제 두께로 스케일 계산
                    depth_scale = real_thickness / depth_variation
                    
                    individual_scales.append({
                        "object_name": class_name,
                        "real_thickness_cm": real_thickness,
                        "depth_variation": depth_variation,
                        "depth_scale_cm_per_unit": depth_scale,
                        "confidence": ref_obj.get("confidence", 0.5)
                    })
            
            if not individual_scales:
                return {
                    "has_scale": False,
                    "method": "insufficient_reference_info",
                    "depth_scale_cm_per_unit": 0.0,
                    "confidence": 0.0
                }
            
            # 신뢰도 가중 평균으로 최종 스케일 계산
            weighted_sum = 0.0
            total_confidence = 0.0
            
            for scale_info in individual_scales:
                confidence = scale_info["confidence"]
                scale = scale_info["depth_scale_cm_per_unit"]
                
                weighted_sum += scale * confidence
                total_confidence += confidence
            
            if total_confidence > 0:
                final_scale = weighted_sum / total_confidence
                final_confidence = min(total_confidence / len(individual_scales), 1.0)
                
                return {
                    "has_scale": True,
                    "method": "reference_object_based",
                    "depth_scale_cm_per_unit": final_scale,
                    "confidence": final_confidence,
                    "reference_count": len(individual_scales),
                    "individual_scales": individual_scales
                }
            else:
                return {
                    "has_scale": False,
                    "method": "zero_confidence",
                    "depth_scale_cm_per_unit": 0.0,
                    "confidence": 0.0
                }
            
        except Exception as e:
            logging.error(f"깊이 스케일 계산 오류: {e}")
            return {
                "has_scale": False,
                "method": "calculation_error",
                "depth_scale_cm_per_unit": 0.0,
                "confidence": 0.0,
                "error": str(e)
            } 

    def _calculate_fallback_info(self, food_features: List[Dict], reference_features: List[Dict], 
                                depth_scale_info: Dict, focal_length_info: Dict) -> Dict:
        """
        기준 객체가 없을 때의 대안적 계산 정보 생성
        
        Args:
            food_features: 음식 특징
            reference_features: 기준 물체 특징
            depth_scale_info: 깊이 스케일 정보
            focal_length_info: 카메라 초점거리 정보
            
        Returns:
            대안적 계산 정보
        """
        try:
            has_reference = len(reference_features) > 0
            has_depth_scale = depth_scale_info.get('has_scale', False)
            has_camera_info = focal_length_info and focal_length_info.get("has_focal_length")
            
            if has_reference and has_depth_scale:
                # 최적 - 기준 물체 기반 계산
                return {
                    "method": "reference_based",
                    "confidence": 0.85,
                    "description": "기준 물체와 깊이 스케일을 활용한 정확한 계산 가능",
                    "recommended_approach": "기준 물체 기반 계산"
                }
            elif has_camera_info:
                # 중간 - 카메라 기반 계산
                focal_length_35mm = focal_length_info.get("focal_length_35mm", 0)
                camera_type = focal_length_info.get("camera_type", "unknown")
                
                # 카메라 타입별 대안 계산 방법
                if camera_type == "smartphone":
                    estimated_distance = self._estimate_smartphone_distance(food_features, focal_length_35mm)
                    pixel_scale = self._calculate_pixel_scale_from_camera(focal_length_35mm, estimated_distance)
                    
                    return {
                        "method": "camera_based",
                        "confidence": 0.65,
                        "description": f"스마트폰 카메라 ({focal_length_35mm}mm) 기반 계산",
                        "recommended_approach": "초점거리 기반 계산",
                        "estimated_distance_cm": estimated_distance,
                        "pixel_scale_cm_per_pixel": pixel_scale,
                        "calculation_notes": [
                            "일반적인 음식 촬영 거리 추정 사용",
                            "스마트폰 센서 특성 고려",
                            "핀홀 카메라 모델 근사치 적용"
                        ]
                    }
                else:
                    return {
                        "method": "camera_based",
                        "confidence": 0.55,
                        "description": f"카메라 기반 계산 (타입: {camera_type})",
                        "recommended_approach": "초점거리 기반 계산",
                        "calculation_notes": [
                            "카메라 타입 불명으로 추정 정확도 제한",
                            "일반적인 카메라 특성 가정"
                        ]
                    }
            else:
                # 최후 - 경험적 추정
                return {
                    "method": "empirical",
                    "confidence": 0.35,
                    "description": "경험적 추정 (기준 물체 및 카메라 정보 없음)",
                    "recommended_approach": "경험적 추정",
                    "fallback_assumptions": [
                        "일반적인 음식 포션 크기 참조",
                        "픽셀 면적 기반 상대적 크기 추정",
                        "표준 음식 밀도 데이터베이스 활용",
                        "보수적 추정 권장"
                    ],
                    "calculation_notes": [
                        "신뢰도 낮음 (0.3-0.5 권장)",
                        "과소추정 경향 고려",
                        "불확실성 큼"
                    ]
                }
            
        except Exception as e:
            logging.error(f"대안적 계산 정보 생성 오류: {e}")
            return {
                "method": "error",
                "confidence": 0.1,
                "description": "계산 정보 생성 실패",
                "recommended_approach": "기본값 사용"
            }
    
    def _estimate_smartphone_distance(self, food_features: List[Dict], focal_length_35mm: float) -> float:
        """
        스마트폰으로 음식 촬영 시 예상 거리 추정
        
        Args:
            food_features: 음식 특징
            focal_length_35mm: 35mm 환산 초점거리
            
        Returns:
            예상 촬영 거리 (cm)
        """
        try:
            if not food_features:
                return 40.0  # 기본값
            
            # 픽셀 면적 기반 거리 추정
            pixel_area = food_features[0].get("pixel_area", 10000)
            
            # 초점거리별 일반적인 촬영 거리
            if focal_length_35mm <= 24:  # 광각
                base_distance = 25.0
            elif focal_length_35mm <= 35:  # 표준
                base_distance = 35.0
            else:  # 망원
                base_distance = 50.0
            
            # 픽셀 면적 기반 보정
            if pixel_area > 50000:  # 큰 음식 or 가까운 거리
                distance_factor = 0.8
            elif pixel_area > 20000:  # 중간 크기
                distance_factor = 1.0
            else:  # 작은 음식 or 먼 거리
                distance_factor = 1.3
            
            estimated_distance = base_distance * distance_factor
            
            # 합리적 범위로 제한
            return max(20.0, min(80.0, estimated_distance))
            
        except Exception as e:
            logging.error(f"스마트폰 거리 추정 오류: {e}")
            return 40.0  # 기본값
    
    def _calculate_pixel_scale_from_camera(self, focal_length_35mm: float, distance_cm: float) -> float:
        """
        카메라 정보로부터 픽셀 스케일 계산
        
        Args:
            focal_length_35mm: 35mm 환산 초점거리
            distance_cm: 촬영 거리
            
        Returns:
            픽셀 스케일 (cm/pixel)
        """
        try:
            # 35mm 필름 센서 크기: 36mm x 24mm
            sensor_width_mm = 36.0
            
            # 일반적인 이미지 해상도 가정 (4000x3000 정도)
            image_width_pixels = 4000.0
            
            # 핀홀 카메라 모델: 실제크기 = (픽셀크기 × 거리) / 초점거리
            # 1픽셀이 실제 세계에서 차지하는 크기
            pixel_size_at_distance = (sensor_width_mm / focal_length_35mm) * (distance_cm / 10.0) / image_width_pixels
            
            return pixel_size_at_distance
            
        except Exception as e:
            logging.error(f"픽셀 스케일 계산 오류: {e}")
            return 0.01  # 기본값 (1픽셀 = 0.01cm) 