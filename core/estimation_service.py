import os
import sys
import logging
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# 상위 디렉토리를 패스에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolo_model import YOLOSegmentationModel
from models.midas_model import MiDaSDepthModel
from models.llm_model import LLMMassEstimator
from utils.feature_extraction import FeatureExtractor
from utils.reference_objects import ReferenceObjectManager
from utils.debug_helper import DebugHelper
from pipeline.config import Config

class MassEstimationPipeline:
    """
    음식 질량 추정 파이프라인
    YOLO Segmentation -> MiDaS Depth -> Feature Extraction -> LLM Mass Estimation
    """
    
    def __init__(self, 
                 yolo_model_path: str = None,
                 midas_model_type: str = None,
                 llm_provider: str = None,
                 llm_model_name: str = None,
                 multimodal_model_name: str = None,
                 api_key: str = None,
                 enable_multimodal: bool = None,
                 debug: bool = None):
        """
        파이프라인 초기화
        
        Args:
            yolo_model_path: YOLO 모델 경로
            midas_model_type: MiDaS 모델 타입
            llm_provider: LLM 제공자 ("gemini" 또는 "openai")
            llm_model_name: LLM 모델 이름
            multimodal_model_name: 멀티모달 모델 이름
            api_key: API 키
            enable_multimodal: 멀티모달 검증 활성화 여부
            debug: 디버그 모드 활성화 여부
        """
        # Config에서 기본값 가져오기
        self.yolo_model_path = yolo_model_path or Config.YOLO_MODEL_PATH
        self.midas_model_type = midas_model_type or Config.MIDAS_MODEL_TYPE
        self.llm_provider = llm_provider or Config.LLM_PROVIDER
        self.llm_model_name = llm_model_name or Config.LLM_MODEL_NAME
        self.multimodal_model_name = multimodal_model_name or Config.MULTIMODAL_MODEL_NAME
        self.api_key = api_key or (Config.GEMINI_API_KEY if self.llm_provider == "gemini" else Config.OPENAI_API_KEY)
        self.enable_multimodal = enable_multimodal if enable_multimodal is not None else Config.ENABLE_MULTIMODAL
        self.debug = debug if debug is not None else Config.DEBUG_MODE
        
        # 로깅 설정
        self._setup_logging()
        
        # 모델 초기화
        self.yolo_model = None
        self.midas_model = None
        self.llm_model = None
        self.feature_extractor = None
        self.reference_manager = None
        
        # 모델 로드
        self._load_models()
        
        # 결과 히스토리
        self.estimation_history = []
        
    def _setup_logging(self):
        """로깅 설정"""
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('mass_estimation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_models(self):
        """모델들 로드"""
        try:
            self.logger.info("모델 로딩 시작...")
            
            # YOLO 모델 로드
            self.yolo_model = YOLOSegmentationModel(self.yolo_model_path)
            self.logger.info("YOLO 모델 로드 완료")
            
            # MiDaS 모델 로드
            self.midas_model = MiDaSDepthModel(self.midas_model_type)
            self.logger.info("MiDaS 모델 로드 완료")
            
            # 유틸리티 초기화 (LLM 모델보다 먼저)
            self.feature_extractor = FeatureExtractor()
            self.reference_manager = ReferenceObjectManager()
            self.debug_helper = DebugHelper(enable_debug=self.debug, simple_mode=Config.SIMPLE_DEBUG)
            
            # LLM 모델 로드 (debug_helper 전달)
            self.llm_model = LLMMassEstimator(
                provider=self.llm_provider,
                model_name=self.llm_model_name,
                multimodal_model=self.multimodal_model_name,
                api_key=self.api_key,
                debug_helper=self.debug_helper
            )
            self.logger.info("LLM 모델 로드 완료")
            
            self.logger.info("모든 모델 로딩 완료")
            
        except Exception as e:
            self.logger.error(f"모델 로딩 실패: {e}")
            raise
    
    def estimate_mass(self, image_path: str, save_results: bool = True) -> Dict:
        """
        이미지에서 음식 질량 추정
        
        Args:
            image_path: 입력 이미지 경로
            save_results: 결과 저장 여부
            
        Returns:
            질량 추정 결과 딕셔너리
        """
        try:
            self.logger.info(f"질량 추정 시작: {image_path}")
            start_time = datetime.now()
            
            # 1단계: YOLO 세그멘테이션
            self.debug_helper.log_step_start("YOLO 세그멘테이션")
            self.logger.info("1단계: YOLO 세그멘테이션 실행")
            segmentation_results = self.yolo_model.segment_image(image_path)
            self.debug_helper.log_segmentation_debug(segmentation_results)
            
            # 디버그 모드일 때 세그멘테이션 시각화
            if self.debug:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                vis_path = f"results/segmentation_{base_name}_{timestamp}.jpg"
                
                self.logger.info(f"세그멘테이션 시각화 생성: {vis_path}")
                self.yolo_model.visualize_segmentation(
                    image_path=image_path,
                    save_path=vis_path,
                    show_masks=True,
                    show_boxes=True,
                    alpha=0.4
                )
                
            self.debug_helper.log_step_end("YOLO 세그멘테이션")
            
            # 2단계: MiDaS 깊이 추정
            self.debug_helper.log_step_start("MiDaS 깊이 추정")
            self.logger.info("2단계: MiDaS 깊이 추정 실행")
            depth_results = self.midas_model.estimate_depth(image_path)
            self.debug_helper.log_depth_debug(depth_results)
            
            # 디버그 모드일 때 깊이 맵 시각화
            if self.debug:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                depth_vis_path = f"results/depth_{base_name}_{timestamp}.jpg"
                
                self.logger.info(f"깊이 맵 시각화 생성: {depth_vis_path}")
                self.midas_model.visualize_depth(
                    image_path=image_path,
                    save_path=depth_vis_path,
                    segmentation_results=segmentation_results,
                    show_colorbar=True,
                    show_stats=True
                )
                
            self.debug_helper.log_step_end("MiDaS 깊이 추정")
            
            # 3단계: 특징 추출 (카메라 정보 포함)
            self.debug_helper.log_step_start("특징 추출 및 카메라 정보 분석")
            self.logger.info("3단계: 특징 추출 실행")
            features = self.feature_extractor.extract_features(segmentation_results, depth_results, image_path)
            
            # 상세 디버그 정보
            self.debug_helper.log_camera_debug(features.get('focal_length_info'))
            self.debug_helper.log_depth_scale_debug(features.get('depth_scale_info'))
            self.debug_helper.log_fallback_info_debug(features.get('fallback_info'))
            self.debug_helper.log_features_debug(features)
            self.debug_helper.log_step_end("특징 추출 및 카메라 정보 분석")
            
            # 4단계: 기준 물체 분석
            self.debug_helper.log_step_start("기준 물체 분석")
            self.logger.info("4단계: 기준 물체 분석 실행")
            reference_analysis = self._analyze_reference_objects(features)
            self.debug_helper.log_step_end("기준 물체 분석")
            
            # 5단계: LLM 질량 추정
            self.debug_helper.log_step_start("LLM 질량 추정")
            self.logger.info("5단계: LLM 질량 추정 실행")
            initial_estimate = self.llm_model.estimate_mass_from_features(features)
            self.debug_helper.log_step_end("LLM 질량 추정")
            
            # 6단계: 멀티모달 검증 (조건부 실행)
            final_estimate = initial_estimate
            if self._should_perform_multimodal_verification(initial_estimate):
                try:
                    self.debug_helper.log_step_start("멀티모달 검증")
                    self.logger.info("6단계: 멀티모달 검증 실행")
                    multimodal_result = self.llm_model.multimodal_verification(image_path, initial_estimate)
                    
                    # 차이 기반 적응형 보정 적용
                    final_estimate = self._apply_adaptive_correction(initial_estimate, multimodal_result)
                    
                    self.debug_helper.log_step_end("멀티모달 검증")
                except Exception as e:
                    self.logger.warning(f"멀티모달 검증 실패: {e}")
                    # 빠른 실패 - 초기 추정값 사용
                    final_estimate = self._create_fallback_estimate(initial_estimate)
            else:
                self.logger.info("멀티모달 검증 건너뜀 (조건 불만족)")
                # 멀티모달 검증 없이 초기 추정값 사용
                final_estimate = self._create_final_estimate_from_initial(initial_estimate)
            
            # 결과 생성
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                "image_path": image_path,
                "processing_time": processing_time,
                "segmentation_results": segmentation_results,
                "depth_results": depth_results,
                "features": features,
                "reference_analysis": reference_analysis,
                "initial_estimate": initial_estimate,
                "final_estimate": final_estimate,
                "timestamp": end_time.isoformat(),
                "pipeline_version": Config.PIPELINE_VERSION
            }
            
            # 결과 저장
            if save_results:
                self._save_results(result)
            
            # 히스토리에 추가
            self.estimation_history.append(result)
            
            # 최종 결과 디버그 출력
            self.debug_helper.log_summary_debug(result)
            
            self.logger.info(f"질량 추정 완료: {final_estimate.get('final_mass', 'N/A')}g")
            
            return result
            
        except Exception as e:
            self.logger.error(f"질량 추정 중 오류: {e}")
            raise
    
    def _analyze_reference_objects(self, features: Dict) -> Dict:
        """기준 물체 분석"""
        try:
            reference_objects = features.get("reference_objects", [])
            
            if not reference_objects:
                return {
                    "has_reference": False,
                    "suggested_objects": self.reference_manager.suggest_reference_objects([]),
                    "confidence": 0.0
                }
            
            # 최적 기준 물체 선택
            best_references = self.reference_manager.get_best_reference_objects(reference_objects)
            
            # 스케일 팩터 계산
            scale_factors = []
            for ref_obj in reference_objects:
                scale_factor = self.reference_manager.calculate_scale_factor(
                    ref_obj, ref_obj["class_name"]
                )
                if scale_factor:
                    scale_factors.append(scale_factor)
            
            # 평균 스케일 팩터
            avg_scale = sum(scale_factors) / len(scale_factors) if scale_factors else Config.DEFAULT_PIXEL_PER_CM
            
            return {
                "has_reference": True,
                "reference_objects": reference_objects,
                "best_references": best_references,
                "scale_factors": scale_factors,
                "average_scale": avg_scale,
                "confidence": min(Config.MAXIMUM_CONFIDENCE_THRESHOLD, len(scale_factors) / Config.SCALE_FACTOR_CONFIDENCE_DIVISOR)  # 설정값 이상이면 최대 신뢰도
            }
            
        except Exception as e:
            self.logger.error(f"기준 물체 분석 오류: {e}")
            return {"has_reference": False, "confidence": 0.0}
    
    def _save_results(self, result: Dict):
        """결과 저장"""
        try:
            # 결과 디렉토리 생성
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mass_estimation_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # 결과 저장 (numpy 배열 제외)
            save_result = self._prepare_result_for_save(result)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"결과 저장 완료: {filepath}")
            
        except Exception as e:
            self.logger.error(f"결과 저장 오류: {e}")
    
    def _prepare_result_for_save(self, result: Dict) -> Dict:
        """저장을 위한 결과 전처리 (numpy 배열 제거)"""
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        save_result = convert_numpy(result.copy())
        
        # 큰 데이터는 제거하고 요약만 저장
        if "segmentation_results" in save_result:
            seg_results = save_result["segmentation_results"]
            for obj_list in ["food_objects", "reference_objects", "all_objects"]:
                if obj_list in seg_results:
                    for obj in seg_results[obj_list]:
                        if "mask" in obj:
                            obj["mask_shape"] = obj["mask"].shape if hasattr(obj["mask"], 'shape') else "N/A"
                            del obj["mask"]  # 마스크 데이터 제거
        
        if "depth_results" in save_result:
            depth_results = save_result["depth_results"]
            if "depth_map" in depth_results:
                depth_results["depth_map_shape"] = depth_results["depth_map"].shape if hasattr(depth_results["depth_map"], 'shape') else "N/A"
                del depth_results["depth_map"]  # 깊이 맵 데이터 제거
        
        return save_result
    
    def batch_estimate(self, image_paths: List[str], save_results: bool = True) -> List[Dict]:
        """
        여러 이미지에 대한 배치 질량 추정
        
        Args:
            image_paths: 이미지 경로들 리스트
            save_results: 결과 저장 여부
            
        Returns:
            질량 추정 결과들 리스트
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                self.logger.info(f"배치 처리 {i+1}/{len(image_paths)}: {image_path}")
                result = self.estimate_mass(image_path, save_results)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"배치 처리 오류 {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def get_estimation_summary(self, result: Dict) -> Dict:
        """추정 결과 요약"""
        try:
            final_estimate = result.get("final_estimate", {})
            features = result.get("features", {})
            reference_analysis = result.get("reference_analysis", {})
            
            # 음식 정보
            food_objects = features.get("food_objects", [])
            food_info = []
            for food in food_objects:
                food_info.append({
                    "name": food.get("class_name", "unknown"),
                    "confidence": food.get("confidence", 0.0),
                    "pixel_area": food.get("pixel_area", 0)
                })
            
            # 기준 물체 정보
            reference_objects = features.get("reference_objects", [])
            reference_info = []
            for ref in reference_objects:
                reference_info.append({
                    "name": ref.get("class_name", "unknown"),
                    "confidence": ref.get("confidence", 0.0),
                    "pixel_area": ref.get("pixel_area", 0)
                })
            
            return {
                "estimated_mass": final_estimate.get("final_mass", 0.0),
                "confidence": final_estimate.get("confidence", 0.0),
                "processing_time": result.get("processing_time", 0.0),
                "food_detected": len(food_info),
                "reference_detected": len(reference_info),
                "food_info": food_info,
                "reference_info": reference_info,
                "has_reference": reference_analysis.get("has_reference", False),
                "reasoning": final_estimate.get("reasoning", ""),
                "method": final_estimate.get("method", "unknown")
            }
            
        except Exception as e:
            self.logger.error(f"요약 생성 오류: {e}")
            return {"error": str(e)}
    
    def visualize_results(self, result: Dict, save_path: str = None) -> str:
        """결과 시각화"""
        try:
            image_path = result["image_path"]
            
            # 세그멘테이션 시각화
            if save_path:
                seg_vis_path = save_path.replace(".jpg", "_segmentation.jpg")
                self.yolo_model.visualize_segmentation(image_path, seg_vis_path)
            
            # 깊이 맵 시각화
            if save_path:
                depth_vis_path = save_path.replace(".jpg", "_depth.jpg")
                self.midas_model.visualize_depth(image_path, depth_vis_path)
            
            return "시각화 완료"
            
        except Exception as e:
            self.logger.error(f"시각화 오류: {e}")
            return f"시각화 오류: {e}"
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "yolo_model": self.yolo_model.get_model_info() if self.yolo_model else None,
            "midas_model": self.midas_model.get_model_info() if self.midas_model else None,
            "llm_model": self.llm_model.get_model_info() if self.llm_model else None,
            "reference_manager": self.reference_manager.get_statistics() if self.reference_manager else None
        }
    
    def get_pipeline_statistics(self) -> Dict:
        """파이프라인 통계 반환"""
        if not self.estimation_history:
            return {"message": "추정 히스토리가 없습니다"}
        
        # 처리 시간 통계
        processing_times = [r.get("processing_time", 0) for r in self.estimation_history]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # 신뢰도 통계
        confidences = []
        for result in self.estimation_history:
            final_estimate = result.get("final_estimate", {})
            confidence = final_estimate.get("confidence", 0.0)
            confidences.append(confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # 질량 범위
        masses = []
        for result in self.estimation_history:
            final_estimate = result.get("final_estimate", {})
            mass = final_estimate.get("final_mass", 0.0)
            if mass > 0:
                masses.append(mass)
        
        return {
            "total_estimations": len(self.estimation_history),
            "average_processing_time": avg_processing_time,
            "average_confidence": avg_confidence,
            "mass_range": {"min": min(masses), "max": max(masses)} if masses else None,
            "successful_estimations": len([r for r in self.estimation_history if not r.get("error")])
        }
    
    def validate_input(self, image_path: str) -> Tuple[bool, str]:
        """입력 이미지 유효성 검사"""
        try:
            # 파일 존재 확인
            if not os.path.exists(image_path):
                return False, f"이미지 파일이 존재하지 않습니다: {image_path}"
            
            # 파일 형식 확인
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            _, ext = os.path.splitext(image_path.lower())
            if ext not in valid_extensions:
                return False, f"지원되지 않는 이미지 형식입니다: {ext}"
            
            # 파일 크기 확인
            file_size = os.path.getsize(image_path)
            if file_size > 50 * 1024 * 1024:  # 50MB 제한
                return False, "이미지 파일 크기가 너무 큽니다 (50MB 제한)"
            
            return True, ""
            
        except Exception as e:
            return False, f"입력 검증 오류: {e}" 

    def _should_perform_multimodal_verification(self, initial_estimate: Dict) -> bool:
        """멀티모달 검증 수행 여부를 판단 - 항상 실행으로 변경"""
        try:
            # 멀티모달 검증이 비활성화된 경우에만 건너뜀
            if not self.enable_multimodal:
                return False
            
            # 멀티모달이 활성화되어 있으면 항상 실행
            self.logger.info("멀티모달 검증 실행 (정확도 향상을 위해 항상 실행)")
            return True
            
        except Exception as e:
            self.logger.error(f"멀티모달 검증 조건 판단 오류: {e}")
            return self.enable_multimodal  # 기본적으로 설정값 따름
    
    def _create_fallback_estimate(self, initial_estimate: Dict) -> Dict:
        """멀티모달 검증 실패 시 폴백 추정값 생성"""
        try:
            return {
                "final_mass": initial_estimate.get("estimated_mass", Config.DEFAULT_MASS),
                "confidence": initial_estimate.get("confidence", Config.DEFAULT_CONFIDENCE) * 0.9,  # 신뢰도 약간 감소
                "reasoning": f"멀티모달 검증 실패, 초기 추정값 사용: {initial_estimate.get('reasoning', '알 수 없음')}",
                "adjustment": "none",
                "method": "fallback_initial",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"폴백 추정값 생성 오류: {e}")
            return {
                "final_mass": Config.DEFAULT_MASS,
                "confidence": Config.DEFAULT_CONFIDENCE,
                "reasoning": "폴백 추정값 생성 실패",
                "adjustment": "none",
                "method": "fallback_default",
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_final_estimate_from_initial(self, initial_estimate: Dict) -> Dict:
        """초기 추정값에서 최종 추정값 생성"""
        try:
            return {
                "final_mass": initial_estimate.get("estimated_mass", Config.DEFAULT_MASS),
                "confidence": initial_estimate.get("confidence", Config.DEFAULT_CONFIDENCE),
                "reasoning": f"초기 추정값 사용: {initial_estimate.get('reasoning', '알 수 없음')}",
                "adjustment": "none",
                "method": "initial_only",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"최종 추정값 생성 오류: {e}")
            return {
                "final_mass": Config.DEFAULT_MASS,
                "confidence": Config.DEFAULT_CONFIDENCE,
                "reasoning": "최종 추정값 생성 실패",
                "adjustment": "none",
                "method": "final_default",
                "timestamp": datetime.now().isoformat()
            } 

    def _apply_adaptive_correction(self, initial_estimate: Dict, multimodal_result: Dict) -> Dict:
        """차이 기반 적응형 보정 적용"""
        try:
            # 초기 추정값과 멀티모달 결과 추출
            initial_mass = initial_estimate.get("estimated_mass", Config.DEFAULT_MASS)
            initial_confidence = initial_estimate.get("confidence", Config.DEFAULT_CONFIDENCE)
            
            multimodal_mass = multimodal_result.get("final_mass", initial_mass)
            multimodal_confidence = multimodal_result.get("confidence", initial_confidence)
            adjustment_type = multimodal_result.get("adjustment", "적절")
            product_info = multimodal_result.get("product_info", "일반음식")
            
            # 차이 비율 계산
            if initial_mass > 0:
                difference_ratio = abs(initial_mass - multimodal_mass) / initial_mass
            else:
                difference_ratio = 1.0
                
            # 차이 기반 적응형 보정 적용
            if difference_ratio < 0.3:  # 30% 이하 차이 - 가중평균 사용
                total_weight = initial_confidence + multimodal_confidence
                if total_weight > 0:
                    final_mass = (initial_mass * initial_confidence + 
                                multimodal_mass * multimodal_confidence) / total_weight
                    final_confidence = min(0.95, (initial_confidence + multimodal_confidence) / 2)
                else:
                    final_mass = (initial_mass + multimodal_mass) / 2
                    final_confidence = 0.5
                
                correction_method = "weighted_average"
                correction_reason = f"차이 {difference_ratio:.1%} - 가중평균 적용"
                
            elif difference_ratio < 0.7:  # 30-70% 차이 - 높은 신뢰도 우선, 낮은 신뢰도로 보정
                if multimodal_confidence > initial_confidence:
                    final_mass = multimodal_mass * 0.7 + initial_mass * 0.3
                    final_confidence = multimodal_confidence * 0.9
                else:
                    final_mass = initial_mass * 0.7 + multimodal_mass * 0.3
                    final_confidence = initial_confidence * 0.9
                
                correction_method = "high_confidence_priority"
                correction_reason = f"차이 {difference_ratio:.1%} - 높은 신뢰도 우선"
                
            else:  # 70% 이상 차이 - 더 높은 신뢰도 선택
                if multimodal_confidence > initial_confidence:
                    final_mass = multimodal_mass
                    final_confidence = multimodal_confidence
                    correction_method = "multimodal_selected"
                else:
                    final_mass = initial_mass
                    final_confidence = initial_confidence * 0.9
                    correction_method = "initial_selected"
                
                correction_reason = f"차이 {difference_ratio:.1%} - 높은 신뢰도 선택"
            
            # 공산품 처리 로직
            if "브랜드" in product_info or "제품명" in product_info:
                # 브랜드/중량이 불분명한 경우 - 초기 추정값 우선하되 멀티모달로 보완
                if "불분명" in product_info or "불명" in product_info:
                    # 초기 추정값을 기준으로 멀티모달 결과로 보완
                    if difference_ratio < 0.5:  # 50% 이하 차이 - 가중평균 (초기값 더 높은 비중)
                        final_mass = initial_mass * 0.7 + multimodal_mass * 0.3
                        final_confidence = min(0.85, initial_confidence * 0.9)
                        correction_method = "uncertain_product_blend"
                        correction_reason = "공산품(불분명) - 초기 추정값 우선하되 멀티모달로 보완"
                    else:  # 50% 이상 차이 - 초기값 우선하되 약간 조정
                        final_mass = initial_mass * 0.8 + multimodal_mass * 0.2
                        final_confidence = min(0.80, initial_confidence * 0.85)
                        correction_method = "uncertain_product_initial"
                        correction_reason = "공산품(불분명) - 초기 추정값 우선 (큰 차이로 인한 약간 조정)"
                
                # 브랜드/중량이 명확한 공산품 - 멀티모달 우선 (기존 로직)
                else:
                    if adjustment_type == "적절":
                        final_mass = multimodal_mass
                        final_confidence = min(0.95, multimodal_confidence)
                        correction_method = "product_priority"
                        correction_reason = "공산품 - 멀티모달 우선"
                    elif adjustment_type == "높음":
                        final_mass = multimodal_mass
                        final_confidence = min(0.90, multimodal_confidence)
                        correction_method = "product_priority"
                        correction_reason = "공산품 - 멀티모달 우선 (높음 조정)"
                    else:  # 낮음
                        final_mass = multimodal_mass
                        final_confidence = min(0.90, multimodal_confidence)
                        correction_method = "product_priority"
                        correction_reason = "공산품 - 멀티모달 우선 (낮음 조정)"
            
            # 최종 결과 구성
            final_estimate = {
                "final_mass": round(final_mass, 1),
                "confidence": round(final_confidence, 3),
                "reasoning": correction_reason,
                "adjustment": adjustment_type,
                "method": correction_method,
                "product_info": product_info,
                "timestamp": datetime.now().isoformat(),
                
                # 보조 결과 추가
                "multimodal_mass": round(multimodal_mass, 1),
                "multimodal_confidence": round(multimodal_confidence, 3),
                "difference_ratio": round(difference_ratio, 3),
                "correction_method": correction_method,
                "correction_reason": correction_reason
            }
            
            return final_estimate
            
        except Exception as e:
            self.logger.error(f"적응형 보정 적용 오류: {e}")
            # 오류 발생 시 멀티모달 결과 사용
            return {
                "final_mass": multimodal_result.get("final_mass", Config.DEFAULT_MASS),
                "confidence": multimodal_result.get("confidence", Config.DEFAULT_CONFIDENCE),
                "reasoning": f"적응형 보정 오류, 멀티모달 결과 사용: {e}",
                "adjustment": multimodal_result.get("adjustment", "none"),
                "method": "multimodal_fallback",
                "timestamp": datetime.now().isoformat()
            } 