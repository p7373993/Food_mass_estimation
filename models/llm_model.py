import openai
import google.generativeai as genai
import json
import base64
import logging
from typing import Dict, List, Optional
import os
from datetime import datetime
from PIL import Image
from pipeline.config import Config

class LLMMassEstimator:
    """
    LLM을 사용하여 음식 질량을 추정하는 클래스
    """
    
    def __init__(self, provider: str = "gemini", model_name: str = "gemini-1.5-flash", 
                 multimodal_model: str = None, api_key: str = None, debug_helper=None):
        """
        LLM 모델 초기화
        
        Args:
            provider: LLM 제공자 ("gemini" 또는 "openai")
            model_name: 사용할 모델 이름
            multimodal_model: 멀티모달 모델 이름
            api_key: API 키
            debug_helper: 디버그 헬퍼 인스턴스
        """
        self.provider = provider
        self.model_name = model_name
        self.multimodal_model = multimodal_model or model_name
        self.debug_helper = debug_helper
        
        if self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Gemini API 키가 필요합니다. 환경변수 GEMINI_API_KEY를 설정하거나 직접 전달해주세요.")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        elif self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 직접 전달해주세요.")
            openai.api_key = self.api_key
        else:
            raise ValueError(f"지원되지 않는 LLM 제공자: {self.provider}")
        
        # 개선된 음식 밀도 데이터베이스 (g/cm³) - 더 정확한 실측값 기반
        self.food_density_db = {
            # 주식류
            "rice": 1.45,           # 밥 (조리된 상태)
            "fried_rice": 1.35,     # 볶음밥
            "porridge": 1.05,       # 죽
            "bibimbap": 1.25,       # 비빔밥
            
            # 면류
            "noodles": 1.15,        # 국수
            "ramen": 1.08,          # 라면
            "pasta": 1.25,          # 파스타
            "spaghetti": 1.22,      # 스파게티
            "udon": 1.10,           # 우동
            "naengmyeon": 1.05,     # 냉면
            
            # 빵류
            "bread": 0.65,          # 식빵
            "toast": 0.70,          # 토스트
            "bun": 0.60,            # 번
            "bagel": 0.75,          # 베이글
            "croissant": 0.55,      # 크루아상
            
            # 고기류
            "beef": 1.20,           # 쇠고기
            "pork": 1.15,           # 돼지고기
            "chicken": 1.10,        # 닭고기
            "fish": 1.08,           # 생선
            "salmon": 1.05,         # 연어
            "tuna": 1.12,           # 참치
            "meat": 1.18,           # 일반 고기
            
            # 채소류
            "vegetables": 0.85,     # 일반 채소
            "lettuce": 0.95,        # 상추
            "cabbage": 0.90,        # 양배추
            "carrot": 0.88,         # 당근
            "onion": 0.92,          # 양파
            "tomato": 0.93,         # 토마토
            "cucumber": 0.96,       # 오이
            "potato": 0.78,         # 감자
            "sweet_potato": 0.80,   # 고구마
            
            # 과일류
            "apple": 0.85,          # 사과
            "banana": 0.89,         # 바나나
            "orange": 0.87,         # 오렌지
            "grape": 0.90,          # 포도
            "watermelon": 0.92,     # 수박
            "melon": 0.90,          # 멜론
            "strawberry": 0.88,     # 딸기
            "fruits": 0.88,         # 일반 과일
            
            # 유제품
            "milk": 1.03,           # 우유
            "cheese": 1.15,         # 치즈
            "yogurt": 1.05,         # 요구르트
            "butter": 0.91,         # 버터
            "cream": 0.98,          # 크림
            "dairy": 1.05,          # 일반 유제품
            
            # 국물류
            "soup": 1.02,           # 국
            "stew": 1.08,           # 찌개
            "broth": 1.01,          # 육수
            "kimchi_soup": 1.05,    # 김치찌개
            "bean_paste_soup": 1.06, # 된장찌개
            
            # 반찬류
            "kimchi": 1.00,         # 김치
            "pickled_vegetables": 1.05, # 절인 채소
            "seasoned_vegetables": 0.95, # 무침
            "pancake": 0.85,        # 전
            "tofu": 1.10,           # 두부
            
            # 간식류
            "chips": 0.45,          # 감자칩
            "cookies": 0.55,        # 쿠키
            "crackers": 0.60,       # 크래커
            "candy": 1.40,          # 사탕
            "chocolate": 1.25,      # 초콜릿
            "snacks": 0.50,         # 일반 과자
            
            # 음료류
            "water": 1.00,          # 물
            "juice": 1.05,          # 주스
            "soda": 1.03,           # 탄산음료
            "coffee": 1.01,         # 커피
            "tea": 1.00,            # 차
            "beer": 0.99,           # 맥주
            "wine": 0.98,           # 와인
            "drinks": 1.02,         # 일반 음료
            
            # 디저트류
            "cake": 0.75,           # 케이크
            "pie": 0.80,            # 파이
            "ice_cream": 0.52,      # 아이스크림
            "pudding": 1.10,        # 푸딩
            "jelly": 1.05,          # 젤리
            "dessert": 0.80,        # 일반 디저트
            
            # 기타
            "egg": 1.03,            # 계란
            "nuts": 0.60,           # 견과류
            "seeds": 0.55,          # 씨앗류
            "oil": 0.92,            # 기름
            "sauce": 1.10,          # 소스
            "seasoning": 1.20,      # 조미료
            
            # 기본값
            "food": 1.00,           # 일반 음식
            "unknown": 0.90         # 알 수 없는 음식
        }
        
        # 개선된 일반적인 음식 포션 크기 (g) - 한국 음식 기준
        self.typical_portions = {
            # 주식류
            "rice": 200,            # 밥 한 공기
            "fried_rice": 250,      # 볶음밥
            "porridge": 300,        # 죽
            "bibimbap": 350,        # 비빔밥
            
            # 면류
            "noodles": 250,         # 국수
            "ramen": 300,           # 라면
            "pasta": 200,           # 파스타
            "spaghetti": 180,       # 스파게티
            "udon": 280,            # 우동
            "naengmyeon": 400,      # 냉면
            
            # 빵류
            "bread": 60,            # 식빵 한 조각
            "toast": 70,            # 토스트
            "bun": 80,              # 번
            "bagel": 90,            # 베이글
            "croissant": 60,        # 크루아상
            
            # 고기류
            "beef": 120,            # 쇠고기 한 조각
            "pork": 100,            # 돼지고기
            "chicken": 150,         # 닭고기
            "fish": 120,            # 생선
            "salmon": 130,          # 연어
            "tuna": 100,            # 참치
            "meat": 120,            # 일반 고기
            
            # 채소류
            "vegetables": 80,       # 일반 채소
            "lettuce": 50,          # 상추
            "cabbage": 60,          # 양배추
            "carrot": 70,           # 당근
            "onion": 60,            # 양파
            "tomato": 150,          # 토마토 하나
            "cucumber": 100,        # 오이
            "potato": 120,          # 감자 하나
            "sweet_potato": 150,    # 고구마
            
            # 과일류
            "apple": 180,           # 사과 하나
            "banana": 120,          # 바나나 하나
            "orange": 200,          # 오렌지 하나
            "grape": 150,           # 포도 한 송이
            "watermelon": 300,      # 수박 한 조각
            "melon": 200,           # 멜론 한 조각
            "strawberry": 15,       # 딸기 하나
            "fruits": 150,          # 일반 과일
            
            # 기본값
            "food": 100,            # 일반 음식
            "unknown": 80           # 알 수 없는 음식
        }
        
        logging.info(f"LLM 모델이 초기화되었습니다: {self.model_name}")
    
    def estimate_mass_from_features(self, features: Dict) -> Dict:
        """
        특징 정보를 기반으로 음식 질량을 추정합니다.
        여러 음식이 감지된 경우 각각을 개별적으로 처리합니다.
        
        Args:
            features: 추출된 특징 정보
            
        Returns:
            Dict: 질량 추정 결과 (여러 음식인 경우 각각의 결과와 총합)
        """
        food_objects = features.get("food_objects", [])
        
        if not food_objects:
            return {
                "estimated_mass": 0,
                "confidence": 0.1,
                "reasoning": "음식이 감지되지 않았습니다",
                "method": "none",
                "individual_foods": [],
                "total_mass": 0,
                "food_count": 0
            }
        
        # 여러 음식 개별 처리
        individual_results = []
        total_mass = 0
        total_confidence_weighted = 0
        
        for i, food_obj in enumerate(food_objects):
            # 개별 음식에 대한 특징 정보 생성
            individual_features = self._create_individual_food_features(features, food_obj, i)
            
            # 개별 음식 질량 추정
            individual_result = self._estimate_single_food_mass(individual_features)
            individual_results.append(individual_result)
            
            # 총 질량 계산
            mass = individual_result.get("estimated_mass", 0)
            confidence = individual_result.get("confidence", 0)
            total_mass += mass
            total_confidence_weighted += mass * confidence
        
        # 전체 평균 신뢰도 계산 (가중 평균)
        overall_confidence = total_confidence_weighted / total_mass if total_mass > 0 else 0
        
        # 디버그 출력
        if self.debug_helper:
            self.debug_helper.log_multiple_foods_debug(individual_results, total_mass, overall_confidence)
        
        return {
            "estimated_mass": total_mass,
            "confidence": overall_confidence,
            "reasoning": f"{len(food_objects)}개 음식의 개별 질량 추정 결과를 합산",
            "method": "multiple_foods_individual",
            "individual_foods": individual_results,
            "total_mass": total_mass,
            "food_count": len(food_objects)
        }
    
    def _create_individual_food_features(self, original_features: Dict, food_obj: Dict, food_index: int) -> Dict:
        """개별 음식에 대한 특징 정보를 생성합니다."""
        individual_features = original_features.copy()
        
        # 해당 음식만 포함하도록 수정
        individual_features["food_objects"] = [food_obj]
        
        # 상대 크기 정보도 해당 음식에 맞게 조정
        relative_size_info = original_features.get("relative_size_info", [])
        if food_index < len(relative_size_info):
            individual_features["relative_size_info"] = [relative_size_info[food_index]]
        else:
            individual_features["relative_size_info"] = []
        
        return individual_features
    
    def _estimate_single_food_mass(self, features: Dict) -> Dict:
        """단일 음식의 질량을 추정합니다."""
        try:
            if self.debug_helper:
                self.debug_helper.log_single_food_estimation_debug(features)
            
            # 기존 로직 사용
            prompt = self._create_mass_estimation_prompt(features)
            
            if self.debug_helper:
                self.debug_helper.log_llm_prompt_debug(prompt)
            
            response = self._call_gemini_text(prompt)
            
            if self.debug_helper:
                self.debug_helper.log_llm_response_debug(response)
            
            parsed_result = self._parse_mass_estimation_response(response)
            
            return {
                "estimated_mass": parsed_result.get("mass", 0),
                "confidence": parsed_result.get("confidence", 0.5),
                "reasoning": parsed_result.get("reasoning", ""),
                "method": "llm_estimation",
                "food_name": features.get("food_objects", [{}])[0].get("class_name", "unknown")
            }
            
        except Exception as e:
            logging.error(f"단일 음식 질량 추정 중 오류: {str(e)}")
            return {
                "estimated_mass": 100,
                "confidence": 0.3,
                "reasoning": f"추정 중 오류 발생: {str(e)}",
                "method": "fallback",
                "food_name": features.get("food_objects", [{}])[0].get("class_name", "unknown")
            }
    
    def multimodal_verification(self, image_path: str, initial_estimate: Dict) -> Dict:
        """
        멀티모달 LLM을 사용한 질량 추정 보완
        
        Args:
            image_path: 원본 이미지 경로
            initial_estimate: 초기 추정 결과
            
        Returns:
            보완된 질량 추정 결과
        """
        try:
            # 간소화된 멀티모달 프롬프트
            prompt = self._create_multimodal_prompt(initial_estimate)
            
            # 디버그: 멀티모달 프롬프트 출력
            if self.debug_helper:
                print(f"\n🎯 멀티모달 검증 프롬프트:")
                print(f"{'='*50}")
                print(prompt)
                print(f"{'='*50}")
                print(f"프롬프트 길이: {len(prompt)} 문자")
                print(f"검증 대상 이미지: {image_path}")
            
            # 멀티모달 LLM 호출
            if self.provider == "gemini":
                response = self._call_gemini_multimodal(prompt, image_path)
            else:
                response = self._call_openai_multimodal(prompt, image_path)
            
            # 응답 파싱
            result = self._parse_multimodal_response(response)
            
            # 디버그: 멀티모달 응답 출력
            if self.debug_helper:
                print(f"\n🔄 멀티모달 검증 응답:")
                print(f"{'='*50}")
                print(f"원본 응답:")
                print(response)
                print(f"{'='*50}")
                print(f"파싱된 결과:")
                print(f"   - 최종 질량: {result.get('mass', 0)}g")
                print(f"   - 신뢰도: {result.get('confidence', 0):.3f}")
                print(f"   - 조정: {result.get('adjustment', 'N/A')}")
                print(f"   - 제품 정보: {result.get('product_info', 'N/A')}")
                if "reasoning" in result:
                    print(f"   - 검증 근거: {result['reasoning']}")
            
            return {
                "final_mass": result["mass"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"],
                "adjustment": result["adjustment"],
                "product_info": result.get("product_info", "일반음식"),
                "method": "multimodal_verification",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"멀티모달 검증 중 오류: {e}")
            # 멀티모달 실패 시 초기 추정값 반환
            return {
                "final_mass": initial_estimate["estimated_mass"],
                "confidence": initial_estimate["confidence"] * 0.8,
                "reasoning": "멀티모달 검증 실패, 초기 추정값 사용",
                "adjustment": "none",
                "method": "fallback",
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_mass_estimation_prompt(self, features: Dict) -> str:
        """LLM이 직접 계산할 수 있도록 개선된 질량 추정 프롬프트 생성"""
        food_objects = features.get("food_objects", [])
        reference_objects = features.get("reference_objects", [])
        depth_scale_info = features.get("depth_scale_info", {})
        
        if not food_objects:
            return '음식이 감지되지 않았습니다. {"mass": 0, "confidence": 0.1, "reasoning": "음식 감지 실패"}'
        
        # 핵심 정보만 포함
        food = food_objects[0]  # 첫 번째 음식만 사용
        
        # 기준 객체 유무 확인
        has_reference = len(reference_objects) > 0
        has_depth_scale = depth_scale_info.get('has_scale', False)
        
        # 프롬프트 구성
        prompt = f"음식 질량 추정 분석:\n\n"
        
        # 1. 음식 정보
        prompt += f"🍽️ 음식 정보:\n"
        prompt += f"  - 종류: {food['class_name']}\n"
        prompt += f"  - 감지 신뢰도: {food.get('confidence', 0.5):.3f}\n"
        prompt += f"  - 픽셀 면적: {food.get('pixel_area', 0):,}픽셀\n"
        
        # 2. 음식 깊이 정보
        food_depth = food.get('depth_info', {})
        prompt += f"  - 평균 깊이: {food_depth.get('mean_depth', 0.0):.3f}\n"
        prompt += f"  - 깊이 변화량: {food_depth.get('depth_variation', 0.0):.3f}\n"
        
        # 3. 기준 물체 정보 (있는 경우)
        if has_reference:
            ref_obj = reference_objects[0]
            prompt += f"\n📏 기준 물체 정보:\n"
            prompt += f"  - 종류: {ref_obj['class_name']}\n"
            prompt += f"  - 감지 신뢰도: {ref_obj.get('confidence', 0.5):.3f}\n"
            prompt += f"  - 픽셀 면적: {ref_obj.get('pixel_area', 0):,}픽셀\n"
            
            ref_depth = ref_obj.get('depth_info', {})
            prompt += f"  - 평균 깊이: {ref_depth.get('mean_depth', 0.0):.3f}\n"
            prompt += f"  - 깊이 변화량: {ref_depth.get('depth_variation', 0.0):.3f}\n"
            
            # 실제 크기 정보
            real_size = ref_obj.get('real_size', {})
            if real_size:
                prompt += f"  - 실제 크기: {real_size.get('width', 0):.1f}cm × {real_size.get('height', 0):.1f}cm\n"
                prompt += f"  - 실제 면적: {real_size.get('area', 0):.1f}cm²\n"
                prompt += f"  - 실제 두께: {real_size.get('thickness', 0):.1f}cm\n"
        else:
            prompt += f"\n❌ 기준 물체 정보:\n"
            prompt += f"  - 기준 물체가 감지되지 않았습니다.\n"
            prompt += f"  - 대안적 계산 방법을 사용해야 합니다.\n"
        
        # 4. 깊이 스케일 정보
        prompt += f"\n🔍 깊이 스케일 정보:\n"
        if has_depth_scale:
            prompt += f"  - 깊이 스케일: {depth_scale_info.get('depth_scale_cm_per_unit', 0.0):.6f} cm/unit\n"
            prompt += f"  - 계산 신뢰도: {depth_scale_info.get('confidence', 0.0):.3f}\n"
            prompt += f"  - 계산 방법: {depth_scale_info.get('method', 'unknown')}\n"
        else:
            prompt += f"  - 깊이 스케일 없음 (방법: {depth_scale_info.get('method', 'unknown')})\n"
            prompt += f"  - 기준 물체 없이는 정확한 깊이 스케일 계산 불가\n"
        
        # 5. 상대 크기 정보
        relative_info = features.get("relative_size_info", [])
        if relative_info and has_reference:
            prompt += f"\n📊 상대 크기 정보:\n"
            for rel in relative_info:
                prompt += f"  - 픽셀 면적 비율: {rel.get('pixel_area_ratio', 1.0):.3f}\n"
                prompt += f"  - 깊이 비율: {rel.get('depth_ratio', 1.0):.3f}\n"
                if rel.get('real_size_ratio'):
                    prompt += f"  - 실제 크기 비율: {rel.get('real_size_ratio', 1.0):.3f}\n"
        
        # 6. 카메라 정보 (있는 경우)
        focal_length_info = features.get("focal_length_info")
        has_camera_info = focal_length_info and focal_length_info.get("has_focal_length")
        
        if has_camera_info:
            prompt += f"\n📷 카메라 정보:\n"
            focal_length = focal_length_info.get("focal_length_mm", 0)
            focal_length_35mm = focal_length_info.get("focal_length_35mm", 0)
            camera_type = focal_length_info.get("camera_type", "unknown")
            
            prompt += f"  - 카메라: {camera_type}\n"
            if focal_length_35mm > 0:
                prompt += f"  - 초점거리: {focal_length_35mm}mm (35mm 환산)\n"
            elif focal_length > 0:
                prompt += f"  - 초점거리: {focal_length:.1f}mm\n"
        
        # 7. 계산 지시 (기준 객체 유무에 따라 다른 가이드)
        prompt += f"\n🎯 계산 과제:\n"
        prompt += f"위 정보를 바탕으로 음식({food['class_name']})의 질량을 추정하세요.\n\n"
        
        prompt += f"💡 계산 가이드:\n"
        
        if has_reference and has_depth_scale:
            # 기준 물체 기반 계산
            prompt += f"🔸 기준 물체 기반 계산 (정확도 높음):\n"
            prompt += f"1. 픽셀 면적과 깊이 스케일을 사용하여 실제 면적 계산\n"
            prompt += f"2. 깊이 변화량과 깊이 스케일을 사용하여 실제 높이 계산\n"
            prompt += f"3. 면적 × 높이 × 형태보정계수로 부피 계산\n"
            prompt += f"4. 해당 음식의 일반적인 밀도를 고려하여 질량 계산\n"
            prompt += f"5. 기준 물체와의 상대 크기를 고려하여 최종 조정\n"
        elif has_camera_info:
            # 카메라 기반 계산
            prompt += f"🔸 카메라 초점거리 기반 계산 (중간 정확도):\n"
            prompt += f"1. 핀홀 카메라 모델 사용: 실제크기 ≈ (픽셀크기 × 거리) / 초점거리\n"
            prompt += f"2. 평균 깊이값을 상대적 거리로 활용\n"
            prompt += f"3. 스마트폰 카메라 특성 고려 (센서 크기, 화각)\n"
            prompt += f"5. 픽셀 면적을 실제 면적으로 변환\n"
            prompt += f"6. 깊이 변화량을 실제 높이로 변환\n"
            prompt += f"7. 면적 × 높이 × 형태보정계수로 부피 계산\n"
            prompt += f"8. 해당 음식의 일반적인 밀도를 고려하여 질량 계산\n"
        else:
            # 경험적 추정
            prompt += f"🔸 경험적 추정 (낮은 정확도):\n"
            prompt += f"1. 픽셀 면적을 기준으로 상대적 크기 추정\n"
            prompt += f"2. 일반적인 음식 크기 데이터베이스 활용\n"
            prompt += f"3. 깊이 변화량을 상대적 높이로 활용\n"
            prompt += f"4. 해당 음식의 표준 포션 크기 고려\n"
            prompt += f"5. 경험적 보정 계수 적용\n"
            prompt += f"6. 불확실성을 고려한 낮은 신뢰도 설정\n"
        
        # 공통 밀도 추정 가이드 추가
        prompt += f"\n🥘 음식 종류별 밀도 추정 가이드:\n"
        prompt += f"음식 종류를 직접 분석하여 적절한 밀도를 추정 계산식에 적용.\n\n"

        
        # 8. 추가 가이드 (기준 객체 없을 때)
        if not has_reference:
            prompt += f"\n⚠️ 기준 물체 없음 - 추가 고려사항:\n"
            prompt += f"• 일반적인 음식 포션 크기 참고\n"
            prompt += f"• 촬영 각도와 거리 추정\n"
            prompt += f"• 신뢰도를 낮게 설정 (0.3-0.6 권장)\n"
            prompt += f"• 보수적 추정 (과소추정 경향)\n"
        
        # 9. 대안적 계산 정보 (fallback_info 활용)
        fallback_info = features.get("fallback_info", {})
        if fallback_info:
            prompt += f"\n🔧 계산 방법 추천:\n"
            prompt += f"  - 방법: {fallback_info.get('description', 'N/A')}\n"
            prompt += f"  - 추천 접근법: {fallback_info.get('recommended_approach', 'N/A')}\n"
            prompt += f"  - 예상 신뢰도: {fallback_info.get('confidence', 0.5):.2f}\n"
            
            # 카메라 기반 계산일 때 추가 정보
            if fallback_info.get('method') == 'camera_based':
                estimated_distance = fallback_info.get('estimated_distance_cm')
                pixel_scale = fallback_info.get('pixel_scale_cm_per_pixel')
                if estimated_distance and pixel_scale:
                    prompt += f"  - 추정 촬영 거리: {estimated_distance:.1f}cm\n"
                    prompt += f"  - 픽셀 스케일: {pixel_scale:.6f} cm/pixel\n"
                
                calculation_notes = fallback_info.get('calculation_notes', [])
                if calculation_notes:
                    prompt += f"  - 계산 참고사항:\n"
                    for note in calculation_notes:
                        prompt += f"    • {note}\n"
            
            # 경험적 추정일 때 추가 정보
            elif fallback_info.get('method') == 'empirical':
                assumptions = fallback_info.get('fallback_assumptions', [])
                if assumptions:
                    prompt += f"  - 기본 가정:\n"
                    for assumption in assumptions:
                        prompt += f"    • {assumption}\n"
                
                calculation_notes = fallback_info.get('calculation_notes', [])
                if calculation_notes:
                    prompt += f"  - 주의사항:\n"
                    for note in calculation_notes:
                        prompt += f"    • {note}\n"
        
        prompt += f"\n📋 응답 형식:\n"
        prompt += f'{{"mass": <추정질량(g)>, "confidence": <신뢰도(0-1)>, "reasoning": "<계산과정 요약>"}}'
        
        return prompt
    
    def _create_multimodal_prompt(self, initial_estimate: Dict) -> str:
        """공산품 정보 출력 가이드가 포함된 멀티모달 검증 프롬프트"""
        mass = initial_estimate.get('estimated_mass', 100)
        prompt = f"이미지 음식 질량 검증. 초기추정: {mass}g\n"
        prompt += f"이미지를 보고 적절한지 판단.\n"
        prompt += f"공산품(포장식품)인 경우 브랜드명, 제품명, 중량 등 보이는 정보도 함께 출력.\n"
        prompt += f'{{"mass": <질량>, "confidence": <0-1>, "adjustment": "<적절/높음/낮음>", "product_info": "<공산품_정보_또는_일반음식>"}}'
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """간소화된 시스템 프롬프트"""
        return """음식 질량 추정 전문가. 부피와 밀도로 질량 계산. JSON 응답만."""
    
    def _get_multimodal_system_prompt(self) -> str:
        """간소화된 멀티모달 시스템 프롬프트"""
        return """이미지 기반 음식 질량 검증. 시각적 크기 평가. 공산품인 경우 제품 정보 식별. JSON 응답만."""
    
    def _parse_mass_estimation_response(self, response: str) -> Dict:
        """간소화된 질량 추정 응답 파싱"""
        try:
            # JSON 부분 추출
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("JSON 형식을 찾을 수 없습니다")
            
            json_str = response[json_start:json_end]
            
            # 제어 문자 제거
            import re
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
            
            result = json.loads(json_str)
            
            # 핵심 필드만 추출
            mass = result.get("mass", Config.DEFAULT_MASS)
            confidence = result.get("confidence", Config.DEFAULT_CONFIDENCE)
            reasoning = result.get("reasoning", "추정 근거 없음")
            
            # 안전한 형변환
            try:
                mass = float(mass)
            except (ValueError, TypeError):
                mass = Config.DEFAULT_MASS
            
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))  # 0-1 범위 제한
            except (ValueError, TypeError):
                confidence = Config.DEFAULT_CONFIDENCE
            
            return {
                "mass": mass,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logging.error(f"응답 파싱 오류: {e}")
            return {
                "mass": Config.DEFAULT_MASS,
                "confidence": Config.DEFAULT_CONFIDENCE,
                "reasoning": "응답 파싱 실패, 기본값 사용"
            }
    
    def _parse_multimodal_response(self, response: str) -> Dict:
        """간소화된 멀티모달 응답 파싱"""
        try:
            # JSON 부분 추출
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("JSON 형식을 찾을 수 없습니다")
            
            json_str = response[json_start:json_end]
            
            # 제어 문자 제거
            import re
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
            
            result = json.loads(json_str)
            
            # 핵심 필드만 추출
            mass = result.get("mass", Config.DEFAULT_MASS)
            confidence = result.get("confidence", Config.DEFAULT_CONFIDENCE)
            adjustment = result.get("adjustment", "적절")
            product_info = result.get("product_info", "일반음식")
            
            # 안전한 형변환
            try:
                mass = float(mass)
            except (ValueError, TypeError):
                mass = Config.DEFAULT_MASS
            
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = Config.DEFAULT_CONFIDENCE
            
            return {
                "mass": mass,
                "confidence": confidence,
                "reasoning": f"멀티모달 검증: {adjustment}",
                "adjustment": adjustment,
                "product_info": product_info
            }
            
        except Exception as e:
            logging.error(f"멀티모달 응답 파싱 오류: {e}")
            return {
                "mass": Config.DEFAULT_MASS,
                "confidence": Config.DEFAULT_CONFIDENCE,
                "reasoning": "멀티모달 응답 파싱 실패",
                "adjustment": "적절",
                "product_info": "일반음식"
            }
    
    def _call_gemini_text(self, prompt: str) -> str:
        """Gemini 텍스트 모델 호출"""
        try:
            system_prompt = self._get_system_prompt()
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # 일정한 답이 나오도록 temperature를 0으로 설정
            generation_config = genai.types.GenerationConfig(
                temperature=0,
                top_p=1,
                top_k=1,
                max_output_tokens=1000,
            )
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logging.error(f"Gemini 텍스트 호출 오류: {e}")
            raise
    
    def _call_gemini_multimodal(self, prompt: str, image_path: str) -> str:
        """Gemini 멀티모달 모델 호출"""
        try:
            # 이미지 로드
            image = Image.open(image_path)
            
            # 시스템 프롬프트와 사용자 프롬프트 결합
            system_prompt = self._get_multimodal_system_prompt()
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # 일정한 답이 나오도록 temperature를 0으로 설정
            generation_config = genai.types.GenerationConfig(
                temperature=0,
                top_p=1,
                top_k=1,
                max_output_tokens=1000,
            )
            
            # 멀티모달 모델 호출
            multimodal_model = genai.GenerativeModel(self.multimodal_model)
            response = multimodal_model.generate_content(
                [full_prompt, image],
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logging.error(f"Gemini 멀티모달 호출 오류: {e}")
            raise
    
    def _call_openai_text(self, prompt: str) -> str:
        """OpenAI 텍스트 모델 호출"""
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # 일정한 답이 나오도록 0으로 설정
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI 텍스트 호출 오류: {e}")
            raise
    
    def _call_openai_multimodal(self, prompt: str, image_path: str) -> str:
        """OpenAI 멀티모달 모델 호출"""
        try:
            # 이미지를 base64로 인코딩
            image_base64 = self._encode_image(image_path)
            
            response = openai.chat.completions.create(
                model=self.multimodal_model,
                messages=[
                    {"role": "system", "content": self._get_multimodal_system_prompt()},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0,  # 일정한 답이 나오도록 0으로 설정
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI 멀티모달 호출 오류: {e}")
            raise

    def _encode_image(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"이미지 인코딩 오류: {e}")
            raise
    
    def get_food_density(self, food_name: str) -> float:
        """개선된 음식 밀도 조회 - 스마트 매칭 포함"""
        if not food_name:
            return self.food_density_db.get("food", 1.0)
        
        food_name_lower = food_name.lower().strip()
        
        # 정확한 매칭 먼저 시도
        if food_name_lower in self.food_density_db:
            return self.food_density_db[food_name_lower]
        
        # 부분 매칭 시도
        for key, density in self.food_density_db.items():
            if key in food_name_lower or food_name_lower in key:
                return density
        
        # 카테고리별 매칭
        category_mapping = {
            "rice": ["밥", "쌀", "rice", "fried_rice", "볶음밥"],
            "noodles": ["면", "라면", "국수", "우동", "냉면", "파스타", "스파게티"],
            "bread": ["빵", "토스트", "베이글", "크루아상"],
            "meat": ["고기", "쇠고기", "돼지고기", "닭고기", "육류"],
            "fish": ["생선", "물고기", "연어", "참치", "어류"],
            "vegetables": ["야채", "채소", "상추", "양배추", "당근", "양파", "토마토", "오이"],
            "fruits": ["과일", "사과", "바나나", "오렌지", "포도", "수박", "멜론", "딸기"],
            "soup": ["국", "찌개", "스프", "육수"],
            "snacks": ["과자", "칩", "쿠키", "크래커"]
        }
        
        for category, keywords in category_mapping.items():
            if any(keyword in food_name_lower for keyword in keywords):
                return self.food_density_db.get(category, 1.0)
        
        return self.food_density_db.get("food", 1.0)
    
    def get_typical_portion(self, food_name: str) -> float:
        """개선된 일반적인 포션 크기 조회 - 스마트 매칭 포함"""
        if not food_name:
            return self.typical_portions.get("food", 100)
        
        food_name_lower = food_name.lower().strip()
        
        # 정확한 매칭 먼저 시도
        if food_name_lower in self.typical_portions:
            return self.typical_portions[food_name_lower]
        
        # 부분 매칭 시도
        for key, portion in self.typical_portions.items():
            if key in food_name_lower or food_name_lower in key:
                return portion
        
        # 카테고리별 매칭
        category_mapping = {
            "rice": ["밥", "쌀", "rice", "fried_rice", "볶음밥"],
            "noodles": ["면", "라면", "국수", "우동", "냉면", "파스타", "스파게티"],
            "bread": ["빵", "토스트", "베이글", "크루아상"],
            "meat": ["고기", "쇠고기", "돼지고기", "닭고기", "육류"],
            "fish": ["생선", "물고기", "연어", "참치", "어류"],
            "vegetables": ["야채", "채소", "상추", "양배추", "당근", "양파", "토마토", "오이"],
            "fruits": ["과일", "사과", "바나나", "오렌지", "포도", "수박", "멜론", "딸기"]
        }
        
        for category, keywords in category_mapping.items():
            if any(keyword in food_name_lower for keyword in keywords):
                return self.typical_portions.get(category, 100)
        
        return self.typical_portions.get("food", 100)
    
    def estimate_mass_with_improved_logic(self, food_name: str, volume_cm3: float) -> Dict:
        """개선된 질량 추정 로직"""
        try:
            # 음식별 밀도 조회
            density = self.get_food_density(food_name)
            
            # 일반적인 포션 크기 조회
            typical_portion = self.get_typical_portion(food_name)
            
            # 부피 기반 질량 계산
            volume_based_mass = volume_cm3 * density
            
            # 포션 크기 기반 보정
            # 일반적인 포션 크기와 비교하여 합리성 체크
            portion_ratio = volume_based_mass / typical_portion
            
            # 극단적인 값 보정
            if portion_ratio > 5.0:  # 5배 이상 크면 보정
                corrected_mass = typical_portion * 2.0  # 2배 정도로 보정
                confidence = 0.4
            elif portion_ratio < 0.2:  # 1/5 이하로 작으면 보정
                corrected_mass = typical_portion * 0.5  # 절반 정도로 보정
                confidence = 0.4
            else:
                corrected_mass = volume_based_mass
                confidence = 0.8
            
            return {
                "estimated_mass": corrected_mass,
                "confidence": confidence,
                "density_used": density,
                "typical_portion": typical_portion,
                "volume_used": volume_cm3,
                "portion_ratio": portion_ratio
            }
            
        except Exception as e:
            logging.error(f"개선된 질량 추정 오류: {e}")
            return {
                "estimated_mass": 100.0,
                "confidence": 0.3,
                "density_used": 1.0,
                "typical_portion": 100.0,
                "volume_used": volume_cm3,
                "portion_ratio": 1.0
            }
    
    def calculate_volume_from_area_depth(self, pixel_area: int, depth_info: Dict, 
                                       pixel_per_cm: float = None) -> float:
        """
        픽셀 면적과 깊이 정보로부터 부피 계산
        
        Args:
            pixel_area: 픽셀 면적
            depth_info: 깊이 정보
            pixel_per_cm: 1cm당 픽셀 수 (기준 물체로부터 계산)
            
        Returns:
            추정 부피 (cm³)
        """
        try:
            # pixel_per_cm 기본값 설정
            if pixel_per_cm is None:
                pixel_per_cm = Config.DEFAULT_PIXEL_PER_CM
            
            # 픽셀 면적을 실제 면적으로 변환
            real_area = pixel_area / (pixel_per_cm ** 2)
            
            # 평균 높이 추정 (깊이 변화의 절반으로 가정)
            avg_height = depth_info.get("depth_variation", Config.DEFAULT_DEPTH_VARIATION) / 2
            
            # 부피 계산
            volume = real_area * avg_height
            
            return max(volume, 0.1)  # 최소 0.1cm³
            
        except Exception as e:
            logging.error(f"부피 계산 오류: {e}")
            return Config.DEFAULT_DEPTH_VARIATION  # 기본값
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "multimodal_model": self.multimodal_model,
            "food_density_db": self.food_density_db,
            "typical_portions": self.typical_portions
        } 