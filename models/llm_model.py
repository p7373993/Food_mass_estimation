import logging
import numpy as np
import google.generativeai as genai
from typing import Dict
import json
import re
import cv2
import base64

from config.settings import settings
from utils.base_model import BaseModel

class LLMMassEstimator(BaseModel):
    """
    LLM (Gemini)을 사용하여 질량을 추정하는 래퍼 클래스. (일반화 및 자기 교정 로직 강화)
    """
    
    def get_model_name(self) -> str:
        return "Gemini LLM 모델"
    
    def _initialize_model(self) -> None:
        if not settings.GEMINI_API_KEY:
            self._log_error("API 키가 설정되지 않았습니다", ValueError(".env 파일을 확인해주세요"))
            self._model = None
            return
        
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
            self._log_success(f"로딩 성공: {settings.LLM_MODEL_NAME}")
        except Exception as e:
            self._log_error("설정 실패", e)
            self._model = None
    
    def estimate_mass_from_features(self, features: dict, debug_helper=None) -> dict:
        if self._model is None:
            return {"error": "LLM 모델이 초기화되지 않았습니다."}
        
        food_objects = features.get("food_objects", [])
        if not food_objects:
            # 음식 객체가 없으면 빈 결과 반환 (멀티모달에서 처리)
            return {
                "food_estimations": [],
                "food_count": 0,
                "no_food_detected": True
            }
        
        # 여러 음식에 대해 각각 질량 추정
        food_estimations = []
        
        logging.info(f"총 {len(food_objects)}개의 음식 객체에 대해 질량 추정을 시작합니다.")
        
        for i, food in enumerate(food_objects):
            logging.info(f"음식 {i+1}/{len(food_objects)} 처리 중: 픽셀 면적 {food.get('pixel_area', 0):,}, 신뢰도 {food.get('confidence', 0):.3f}, bbox {food.get('bbox', [])}")
            
            prompt = self._build_prompt_for_food(features, food, i)
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=settings.LLM_TEMPERATURE, 
                        top_p=settings.LLM_TOP_P,
                        candidate_count=1  # 완전히 결정론적
                    ),
                )
                mass_info = self._parse_response(response.text)
                if debug_helper:
                    debug_helper.log_initial_mass_calculation_debug(features, prompt, response.text, mass_info, food_index=i)
                
                # 음식별 정보 추가
                mass_info["food_index"] = i
                mass_info["food_bbox"] = food.get("bbox", [])
                mass_info["food_pixel_area"] = food.get("pixel_area", 0)
                food_estimations.append(mass_info)
                
                logging.info(f"음식 {i+1} 질량 추정 완료: {mass_info.get('estimated_mass_g', 0):.1f}g")
                    
            except Exception as e:
                logging.error(f"음식 {i} 질량 추정 중 오류 발생: {e}")
                food_estimations.append({
                    "error": str(e),
                    "food_index": i,
                    "food_bbox": food.get("bbox", []),
                    "food_pixel_area": food.get("pixel_area", 0)
                })
        
        return {
            "food_estimations": food_estimations,
            "food_count": len(food_objects)
        }

    def verify_mass_with_multimodal(self, image: np.ndarray, initial_estimation: dict, features: dict) -> dict:
        if self._model is None:
            return {"error": "LLM 모델이 초기화되지 않았습니다."}
        try:
            multimodal_model = genai.GenerativeModel(settings.MULTIMODAL_MODEL_NAME or settings.LLM_MODEL_NAME)
            
            # 디버그: 원본 이미지 정보 출력
            if settings.DEBUG_MODE:
                print(f"\n🔍 멀티모달 검증 이미지 처리:")
                print(f"   원본 이미지 크기: {image.shape}")
                print(f"   원본 BGR 평균: {np.mean(image, axis=(0,1))}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            max_size = 1536
            h, w = image_rgb.shape[:2]
            
            if settings.DEBUG_MODE:
                print(f"   RGB 변환 후 크기: {image_rgb.shape}")
                print(f"   RGB 평균: {np.mean(image_rgb, axis=(0,1))}")
            
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_rgb = cv2.resize(image_rgb, (new_w, new_h))
                if settings.DEBUG_MODE:
                    print(f"   리사이즈 후 크기: {image_rgb.shape} (스케일: {scale:.3f})")
                    print(f"   리사이즈 후 RGB 평균: {np.mean(image_rgb, axis=(0,1))}")
            
            # 디버그: 처리된 이미지 저장
            if settings.DEBUG_MODE:
                cv2.imwrite("debug_multimodal_input.jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                print(f"   멀티모달 입력 이미지 저장: debug_multimodal_input.jpg")
            
            # 반드시 BGR로 변환 후 인코딩 (색상 문제 방지)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', image_bgr)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            if settings.DEBUG_MODE:
                print(f"   JPEG 버퍼 크기: {len(buffer)} bytes")
                print(f"   Base64 길이: {len(image_base64)} 문자")
            
            prompt = self._build_multimodal_prompt(initial_estimation, features)
            multimodal_content = [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}]}]
            
            if settings.DEBUG_MODE:
                print(f"   멀티모달 프롬프트 길이: {len(prompt)} 문자")
            
            response = multimodal_model.generate_content(
                multimodal_content,
                generation_config=genai.types.GenerationConfig(
                    temperature=settings.LLM_TEMPERATURE, 
                    top_p=settings.LLM_TOP_P,
                    candidate_count=1  # 완전히 결정론적
                ),
            )
            
            if settings.DEBUG_MODE:
                print(f"   LLM 응답 길이: {len(response.text)} 문자")
                print(f"   LLM 응답 미리보기: {response.text[:200]}...")
            
            return self._parse_multimodal_response(response.text, initial_estimation, features)
        except Exception as e:
            logging.error(f"멀티모달 검증 중 오류 발생: {e}")
            return {"error": str(e)}

    def _build_prompt_for_food(self, features: dict, food: dict, food_index: int) -> str:
        """특정 음식 객체에 대한 프롬프트 생성"""
        reference_objects = features.get("reference_objects", [])
        depth_scale_info = features.get("depth_scale_info", {})
        has_reference = len(reference_objects) > 0
        has_depth_scale = depth_scale_info.get('has_scale', False)
        
        prompt = f"음식 {food_index + 1} 질량 추정 분석:\n\n"
        prompt += f"🍽️ 음식 정보:\n  - 종류: {food.get('class_name', '알수없음')}\n  - 픽셀 면적: {food.get('pixel_area', 0):,}픽셀\n"
        food_depth = food.get('depth_info', {})
        prompt += f"  - 평균 깊이값(상대적): {food_depth.get('mean_depth', 0.0):.3f}\n  - 깊이 변화량(상대적): {food_depth.get('depth_variation', 0.0):.3f}\n"
        
        if has_reference:
            ref_obj = reference_objects[0]
            prompt += f"\n📏 기준 물체 정보:\n  - 종류: {ref_obj.get('class_name')}\n"
            real_size = ref_obj.get('real_size', {})
            if real_size:
                prompt += f"  - 실제 크기: {real_size.get('width', 0):.1f}cm × {real_size.get('height', 0):.1f}cm, 두께: {real_size.get('thickness', 0):.1f}cm\n"
        
        if has_depth_scale:
            prompt += f"\n🔍 계산된 실제 스케일:\n  - 깊이 스케일: {depth_scale_info.get('depth_scale_cm_per_unit', 0.0):.4f} cm/unit\n"
            if depth_scale_info.get('pixel_per_cm2_ratio'):
                prompt += f"  - 면적 비율: {depth_scale_info.get('pixel_per_cm2_ratio'):.2f} pixels/cm²\n"
        
        prompt += "\n🎯 계산 과제:\n위 정보를 바탕으로 이 음식의 질량을 g(그램) 단위로 추정하세요. 부피(cm³)를 먼저 계산한 후, 일반적인 음식 밀도(약 0.8~1.2 g/cm³)를 적용하세요.\n"
        prompt += "\n💡 계산 가이드:\n"
        if has_reference and has_depth_scale and depth_scale_info.get('pixel_per_cm2_ratio'):
            prompt += "1. '픽셀 면적'을 '면적 비율'로 나누어 음식의 실제 면적(cm²)을 계산하세요.\n"
            prompt += "2. '깊이 변화량'과 '깊이 스케일'을 곱하여 음식의 실제 높이(cm)를 추정하세요.\n"
            prompt += "3. 추정된 실제 면적과 높이를 곱해 부피(cm³)를 계산하세요. (형태 보정 계수 0.6 적용)\n"
            prompt += "4. 계산된 부피에 음식의 예상 밀도(g/cm³)를 곱해 최종 질량(g)을 계산하세요.\n"
        else:
            prompt += "정확한 스케일 정보가 없으므로, 음식의 픽셀 면적과 일반적인 음식 크기를 고려하여 경험적으로 추정하세요. 신뢰도를 낮게 설정하세요.\n"
        
        prompt += '\n📋 응답 형식 (JSON):\n{"estimated_mass_g": <추정 질량(g)>, "confidence": <신뢰도(0.0~1.0)>, "reasoning": "<계산 과정 및 근거 요약>"}'
        return prompt.strip()

    def _build_prompt(self, features: dict) -> str:
        food_objects = features.get("food_objects", [])
        if not food_objects:
            return '음식이 감지되지 않았습니다. {"mass": 0, "confidence": 0.1, "reasoning": "음식 감지 실패"}'
        food = food_objects[0]
        reference_objects = features.get("reference_objects", [])
        depth_scale_info = features.get("depth_scale_info", {})
        has_reference = len(reference_objects) > 0
        has_depth_scale = depth_scale_info.get('has_scale', False)
        
        prompt = "음식 질량 추정 분석:\n\n"
        prompt += f"🍽️ 음식 정보:\n  - 종류: {food.get('class_name', '알수없음')}\n  - 픽셀 면적: {food.get('pixel_area', 0):,}픽셀\n"
        food_depth = food.get('depth_info', {})
        prompt += f"  - 평균 깊이값(상대적): {food_depth.get('mean_depth', 0.0):.3f}\n  - 깊이 변화량(상대적): {food_depth.get('depth_variation', 0.0):.3f}\n"
        
        if has_reference:
            ref_obj = reference_objects[0]
            prompt += f"\n📏 기준 물체 정보:\n  - 종류: {ref_obj.get('class_name')}\n"
            real_size = ref_obj.get('real_size', {})
            if real_size:
                prompt += f"  - 실제 크기: {real_size.get('width', 0):.1f}cm × {real_size.get('height', 0):.1f}cm, 두께: {real_size.get('thickness', 0):.1f}cm\n"
        
        if has_depth_scale:
            prompt += f"\n🔍 계산된 실제 스케일:\n  - 깊이 스케일: {depth_scale_info.get('depth_scale_cm_per_unit', 0.0):.4f} cm/unit\n"
            if depth_scale_info.get('pixel_per_cm2_ratio'):
                prompt += f"  - 면적 비율: {depth_scale_info.get('pixel_per_cm2_ratio'):.2f} pixels/cm²\n"
        
        prompt += "\n🎯 계산 과제:\n위 정보를 바탕으로 음식의 질량을 g(그램) 단위로 추정하세요. 부피(cm³)를 먼저 계산한 후, 일반적인 음식 밀도(약 0.8~1.2 g/cm³)를 적용하세요.\n"
        prompt += "\n💡 계산 가이드:\n"
        if has_reference and has_depth_scale and depth_scale_info.get('pixel_per_cm2_ratio'):
            prompt += "1. '픽셀 면적'을 '면적 비율'로 나누어 음식의 실제 면적(cm²)을 계산하세요.\n"
            prompt += "2. '깊이 변화량'과 '깊이 스케일'을 곱하여 음식의 실제 높이(cm)를 추정하세요.\n"
            prompt += "3. 추정된 실제 면적과 높이를 곱해 부피(cm³)를 계산하세요. (형태 보정 계수 0.6 적용)\n"
            prompt += "4. 계산된 부피에 음식의 예상 밀도(g/cm³)를 곱해 최종 질량(g)을 계산하세요.\n"
        else:
            prompt += "정확한 스케일 정보가 없으므로, 음식의 픽셀 면적과 일반적인 음식 크기를 고려하여 경험적으로 추정하세요. 신뢰도를 낮게 설정하세요.\n"
        prompt += '\n📋 응답 형식 (JSON):\n{"estimated_mass_g": <추정 질량(g)>, "confidence": <신뢰도(0.0~1.0)>, "reasoning": "<계산 과정 및 근거 요약>"}'
        return prompt.strip()

    def _parse_response(self, response_text: str) -> dict:
        try:
            json_part = response_text.split('```json')[-1].split('```')[0].strip()
            return json.loads(json_part)
        except Exception as e:
            logging.error(f"LLM 응답 파싱 실패: {e}\n원본 응답: {response_text}")
            return {"error": "LLM 응답을 파싱할 수 없습니다."}

    def _build_multimodal_prompt(self, initial_estimation: dict, features: dict) -> str:
        food_objects = features.get("food_objects", [])
        food_count = len(food_objects)
        no_food_detected = initial_estimation.get("no_food_detected", False)
        
        # 기준물체 신뢰도 정보 추출
        reference_objects = features.get("reference_objects", [])
        reference_confidence = 0.0
        reference_info = ""
        
        if reference_objects:
            ref_obj = reference_objects[0]
            reference_confidence = ref_obj.get("confidence", 0.0)
            reference_info = f"- 기준물체: {ref_obj.get('class_name', '알수없음')} (신뢰도: {reference_confidence:.2f})\n"
        else:
            reference_info = "- 기준물체: 없음 (신뢰도: 0.0)\n"
        
        prompt = f"""
음식 질량 추정 멀티모달 검증:

📊 초기 추정 정보:
- 감지된 음식 개수: {food_count}개
{reference_info}
"""
        
        if no_food_detected:
            prompt += "- YOLO 모델이 음식을 감지하지 못했습니다. 멀티모달 검증으로 최종 판단합니다.\n"
        elif "food_estimations" in initial_estimation:
            for i, est in enumerate(initial_estimation["food_estimations"]):
                if "error" not in est:
                    prompt += f"- 음식 {i+1} 초기 추정 질량: {est.get('estimated_mass_g', 0):.1f}g\n"

        # 기준물체 신뢰도가 낮을 때 멀티모달에게 더 큰 권한 부여
        if reference_confidence < 0.5:
            prompt += f"""
⚠️ 중요: 기준물체 신뢰도가 낮습니다 ({reference_confidence:.2f}). 
이 경우 초기 추정 질량보다 멀티모달의 시각적 판단을 우선시하세요.
초기 추정이 과대 추정된 것 같다면 질량을 줄이고, 과소 추정된 것 같다면 질량을 늘리세요.
"""

        prompt += f"""
🔍 검증 과제:
이미지를 보고 다음 규칙에 따라 단계적으로 분석하세요.

1.  **1차 시각적 식별**:
    - 이미지에 보이는 모든 음식 물체를 있는 그대로 설명하세요
    - 음식이 여러 개 있다면 각각을 구분하여 설명하세요
    - 음식이 보이지 않는다면 "음식이 없음"이라고 명시하세요

2.  **재검토 및 최종 판단**:
    - 1차 식별 결과에 대해 다른 가능성은 없는지 비판적으로 검토하세요.
    - 위 가능성을 고려하여, 1차 판단이 가장 합리적인지, 아니면 더 적절한 다른 음식 이름이 있는지 최종 결론을 내리세요.

3.  **라벨 텍스트 분석**:
    - 라벨이 보이면, 제품명과 중량(g 또는 ml)을 **보이는 그대로 인용**하세요.

4.  **질량 조정 (기준물체 신뢰도가 낮을 때)**:
    - 초기 추정 질량이 현실적으로 보이는지 검토하세요
    - "좀 많다", "과대 추정", "너무 크다" 등의 표현이 있다면 질량을 줄이세요
    - "좀 적다", "과소 추정", "너무 작다" 등의 표현이 있다면 질량을 늘리세요
    - 조정 비율은 보통 10-30% 정도가 적절합니다
    - **중요**: 조정한 최종 질량을 "verified_mass_g" 필드에 정확히 입력하세요

📋 응답 형식 (JSON):
{{
    "foods": [
        {{
            "food_name": "<음식 이름>",
            "quoted_text": {{
                "product_name": "<라벨에서 읽은 제품명 그대로>",
                "weight": "<라벨에서 읽은 중량 그대로>"
            }},
            "verified_mass_g": <이 음식 1개의 질량(g)>,
            "confidence": <신뢰도(0.0~1.0)>,
            "reasoning": "<이 음식에 대한 분석 과정 및 근거>"
        }}
    ],
    "overall_confidence": <전체 신뢰도(0.0~1.0)>,
    "reasoning": "<전체 분석 과정 및 근거>"
}}
"""
        return prompt.strip()

    def _parse_multimodal_response(self, response_text: str, initial_estimation: dict, features: dict) -> dict:
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return self._process_parsed_multimodal_response(parsed, initial_estimation, features)
            else:
                raise ValueError("JSON 응답을 찾을 수 없습니다.")
        except Exception as e:
            logging.error(f"멀티모달 응답 파싱 및 처리 오류: {e}")
            return {"error": f"멀티모달 검증 실패: {e}", "confidence": 0.1, "estimated_mass_g": initial_estimation.get('estimated_mass_g', 0)}

    def _process_parsed_multimodal_response(self, parsed: dict, initial_estimation: dict, features: dict) -> dict:
        foods = parsed.get('foods', [])
        food_verifications = []
        
        # 기준물체 신뢰도 정보 추출
        reference_confidence = 0.0
        reference_objects = features.get("reference_objects", [])
        if reference_objects:
            reference_confidence = reference_objects[0].get("confidence", 0.0)
        
        for i, food_info in enumerate(foods):
            food_name = food_info.get('food_name', '알수없음')
            reasoning = food_info.get('reasoning', '추론 과정 없음')
            confidence = food_info.get('confidence', 0.5)
            
            quoted_text = food_info.get('quoted_text', {})
            quoted_weight_str = quoted_text.get('weight')

            final_mass = None
            verification_method = "visual_estimation"

            # 1. 라벨에서 읽은 무게 정보가 있는지 확인
            if quoted_weight_str:
                mass_match = re.search(r'(\d+(?:\.\d+)?)', quoted_weight_str)
                if mass_match:
                    final_mass = float(mass_match.group(1))
                    confidence = 0.95
                    verification_method = "label_based"
                    logging.info(f"라벨 정보 기반 질량 추정: {final_mass}g")

            # 2. 라벨 정보가 없으면, LLM의 시각적 판단에 의존
            if final_mass is None:
                # 멀티모달이 제안한 질량을 우선 사용
                final_mass = food_info.get('verified_mass_g', 0)
                logging.info(f"멀티모달 제안 질량: {final_mass}g")
                
                # 3. 기준물체 신뢰도가 낮을 때 멀티모달 피드백 강화
                if reference_confidence < 0.5:
                    # 초기 추정 질량과 비교하여 조정
                    initial_mass = 0
                    if "food_estimations" in initial_estimation and i < len(initial_estimation["food_estimations"]):
                        initial_mass = initial_estimation["food_estimations"][i].get("estimated_mass_g", 0)
                    
                    if initial_mass > 0:
                        # 멀티모달이 reasoning에서 조정된 질량을 명시했는지 확인
                        reasoning_lower = reasoning.lower()
                        
                        # "Xg으로 조정" 패턴 찾기
                        adjustment_match = re.search(r'(\d+(?:\.\d+)?)g\s*으로\s*조정', reasoning)
                        if adjustment_match:
                            adjusted_mass = float(adjustment_match.group(1))
                            final_mass = adjusted_mass
                            logging.info(f"기준물체 신뢰도 낮음: 멀티모달이 명시한 조정값 사용 {initial_mass:.1f}g → {final_mass:.1f}g")
                            verification_method = "multimodal_adjusted"
                        # 키워드 기반 자동 조정 (멀티모달이 명시하지 않은 경우)
                        elif any(keyword in reasoning_lower for keyword in ["좀 많", "과대", "너무 크", "많이", "큰 편"]):
                            # 질량을 20-30% 줄임
                            reduction_factor = 0.75  # 25% 감소
                            final_mass = initial_mass * reduction_factor
                            logging.info(f"기준물체 신뢰도 낮음: 키워드 기반 질량 조정 {initial_mass:.1f}g → {final_mass:.1f}g (25% 감소)")
                            verification_method = "multimodal_adjusted"
                        elif any(keyword in reasoning_lower for keyword in ["좀 적", "과소", "너무 작", "적게", "작은 편"]):
                            # 질량을 20-30% 늘림
                            increase_factor = 1.25  # 25% 증가
                            final_mass = initial_mass * increase_factor
                            logging.info(f"기준물체 신뢰도 낮음: 키워드 기반 질량 조정 {initial_mass:.1f}g → {final_mass:.1f}g (25% 증가)")
                            verification_method = "multimodal_adjusted"
                        else:
                            # 피드백이 없으면 멀티모달 추정값 사용
                            logging.info(f"기준물체 신뢰도 낮음: 멀티모달 추정값 사용 {final_mass:.1f}g")
                            verification_method = "multimodal_estimation"
            
            food_verifications.append({
                "food_index": i,
                "food_name": food_name,
                "verified_mass_g": final_mass,
                "confidence": confidence,
                "verification_method": verification_method,
                "quoted_text": quoted_text,
                "reasoning": reasoning
            })
            
        return {
            "food_verifications": food_verifications,
            "overall_confidence": parsed.get('overall_confidence', 0.5),
            "multimodal_estimation": parsed,
            "reasoning": parsed.get('reasoning', '전체 분석 과정 없음')
        }

# 싱글톤 인스턴스 생성
llm_estimator = LLMMassEstimator()