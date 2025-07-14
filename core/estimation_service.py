import logging
import numpy as np
from typing import Dict

# 새로운 모델 래퍼 및 유틸리티 임포트
from models.yolo_model import yolo_model, load_image
from models.midas_model import midas_model
from models.llm_model import llm_estimator
from utils.feature_extraction import FeatureExtractor
from utils.debug_helper import DebugHelper
from config.settings import settings

class MassEstimationService:
    """
    모든 AI 모델과 특징 추출기를 사용하여 질량 추정 파이프라인을 총괄하는 서비스 클래스.
    """
    
    def __init__(self):
        """
        서비스 초기화. 싱글톤으로 생성된 모델 및 유틸리티 인스턴스를 가져옵니다.
        """
        self.yolo_model = yolo_model
        self.midas_model = midas_model
        self.llm_estimator = llm_estimator
        self.feature_extractor = FeatureExtractor()
        
        if settings.DEBUG_MODE:
            logging.basicConfig(level=logging.DEBUG, format=settings.LOG_FORMAT)
        else:
            logging.basicConfig(level=logging.INFO, format=settings.LOG_FORMAT)

    def run_pipeline(self, image: np.ndarray, image_path: str = None) -> Dict:
        """
        전체 질량 추정 파이프라인을 실행합니다.

        Args:
            image (np.ndarray): 처리할 이미지 (OpenCV BGR 형식).
            image_path (str, optional): 원본 이미지 경로 (EXIF 정보 추출용). Defaults to None.

        Returns:
            Dict: 최종 추정 결과.
        """
        try:
            logging.info("질량 추정 파이프라인 시작...")
            
            # DebugHelper 인스턴스 생성
            debug_helper = DebugHelper(
                enable_debug=settings.DEBUG_MODE, 
                simple_mode=settings.SIMPLE_DEBUG,
                image_path=image_path
            )

            # 1단계: YOLO 세그멘테이션
            logging.info("1단계: YOLO 세그멘테이션 실행")
            segmentation_results = self.yolo_model.segment_image(image)
            debug_helper.log_segmentation_debug(segmentation_results)
            
            # 시각화 추가
            debug_helper.save_segmentation_visualization(image, segmentation_results)
            
            # 객체 감지 실패 시 파이프라인 중단
            if not segmentation_results.get("food_objects"):
                logging.warning("음식 객체를 감지하지 못했습니다.")
                raise ValueError("음식 객체를 찾을 수 없습니다.")

            # 2단계: MiDaS 깊이 추정
            logging.info("2단계: MiDaS 깊이 추정 실행")
            depth_map = self.midas_model.estimate_depth(image)
            if depth_map is None:
                logging.error("깊이 추정에 실패했습니다.")
                return {"error": "깊이 추정에 실패했습니다."}

            # 시각화 추가
            debug_helper.save_depth_map_visualization(image, depth_map, segmentation_results)

            # 3단계: 특징 추출
            logging.info("3단계: 특징 추출 실행")
            features = self.feature_extractor.extract_features(segmentation_results, depth_map, image_path)

            # 4단계: LLM 질량 추정
            logging.info("4단계: LLM 질량 추정 실행")
            debug_helper.log_step_start("LLM 질량 추정")
            
            estimated_result = self.llm_estimator.estimate_mass_from_features(
                features, debug_helper=debug_helper
            )
            
            debug_helper.log_step_end("LLM 질량 추정")

            # 5단계: 멀티모달 검증 (설정에 따라 선택적 실행)
            if settings.ENABLE_MULTIMODAL and not estimated_result.get("error"):
                logging.info("5단계: 멀티모달 검증 실행")
                verification_result = self.llm_estimator.verify_mass_with_multimodal(
                    image, estimated_result, features
                )
                
                if not verification_result.get("error"):
                    # 검증 결과로 최종 결과 업데이트
                    estimated_result = verification_result
                    logging.info("멀티모달 검증 완료")
                else:
                    logging.warning(f"멀티모달 검증 실패: {verification_result.get('error')}")
                    logging.info("초기 추정 결과 사용")
            else:
                logging.info("멀티모달 검증 건너뜀 (비활성화 또는 오류)")

            # 최종 결과 조합
            final_result = {
                "mass_estimation": estimated_result,
                "features": self._simplify_features_for_response(features)
            }
            logging.info("질량 추정 파이프라인 성공적으로 완료.")
            return final_result

        except Exception as e:
            logging.error(f"파이프라인 실행 중 심각한 오류 발생: {e}", exc_info=True)
            return {"error": f"서버 내부 오류가 발생했습니다: {e}"}

    def _simplify_features_for_response(self, features: dict) -> dict:
        """API 응답에 포함될 특징 정보를 간소화합니다. (numpy 타입을 Python 타입으로 변환)"""
        
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
        
        simplified = {}
        for key, value in features.items():
            if isinstance(value, list):
                simplified[key] = []
                for item in value:
                    if isinstance(item, dict):
                        # 마스크와 같이 용량이 큰 데이터는 응답에서 제외
                        item.pop("mask", None)
                        # numpy 타입 변환
                        simplified_item = convert_numpy_types(item)
                        simplified[key].append(simplified_item)
            else:
                simplified[key] = convert_numpy_types(value)
        return simplified

# 서비스 인스턴스 생성 (서버에서 이 인스턴스를 사용)
mass_estimation_service = MassEstimationService() 