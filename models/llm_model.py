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
            return {"error": "음식 객체가 감지되지 않았습니다."}
        
        # 여러 음식에 대해 각각 질량 추정
        food_estimations = []
        
        for i, food in enumerate(food_objects):
            prompt = self._build_prompt_for_food(features, food, i)
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=settings.LLM_TEMPERATURE, top_p=settings.LLM_TOP_P
                    ),
                )
                mass_info = self._parse_response(response.text)
                if debug_helper:
                    debug_helper.log_initial_mass_calculation_debug(features, prompt, response.text, mass_info)
                
                # 음식별 정보 추가
                mass_info["food_index"] = i
                mass_info["food_bbox"] = food.get("bbox", [])
                mass_info["food_pixel_area"] = food.get("pixel_area", 0)
                food_estimations.append(mass_info)
                    
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
                    temperature=settings.LLM_TEMPERATURE, top_p=settings.LLM_TOP_P
                ),
            )
            
            if settings.DEBUG_MODE:
                print(f"   LLM 응답 길이: {len(response.text)} 문자")
                print(f"   LLM 응답 미리보기: {response.text[:200]}...")
            
            return self._parse_multimodal_response(response.text, initial_estimation)
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
        
        prompt = f"""
음식 질량 추정 멀티모달 검증:

📊 초기 추정 정보:
- 감지된 음식 개수: {food_count}개
"""
        
        if "food_estimations" in initial_estimation:
            for i, est in enumerate(initial_estimation["food_estimations"]):
                if "error" not in est:
                    prompt += f"- 음식 {i+1} 초기 추정 질량: {est.get('estimated_mass_g', 0):.1f}g\n"

        prompt += f"""
🔍 검증 과제:
이미지를 보고 다음 규칙에 따라 단계적으로 분석하세요.

1.  **1차 시각적 식별**:
    - 이미지에 보이는 모든 음식 물체를 있는 그대로 설명하세요
    - 음식이 여러 개 있다면 각각을 구분하여 설명하세요

2.  **재검토 및 최종 판단**:
    - 1차 식별 결과에 대해 다른 가능성은 없는지 비판적으로 검토하세요.
    - **오류 가능성 체크**:
        - **조명/반사**: 조명이나 포장지 반사 때문에 색상이 왜곡될 수 있습니다. (예: 갈색이 노랗거나 검게 보일 수 있음)
        - **모양의 함정**: 특정 모양이 여러 음식에서 나타날 수 있습니다. (예: 꽃 모양은 과자, 떡, 젤리 등 다양함)
    - 위 가능성을 고려하여, 1차 판단이 가장 합리적인지, 아니면 더 적절한 다른 음식 이름이 있는지 최종 결론을 내리세요.

3.  **라벨 텍스트 분석**:
    - 라벨이 보이면, 제품명과 중량(g 또는 ml)을 **보이는 그대로 인용**하세요.

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

    def _parse_multimodal_response(self, response_text: str, initial_estimation: dict) -> dict:
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return self._process_parsed_multimodal_response(parsed, initial_estimation)
            else:
                raise ValueError("JSON 응답을 찾을 수 없습니다.")
        except Exception as e:
            logging.error(f"멀티모달 응답 파싱 및 처리 오류: {e}")
            return {"error": f"멀티모달 검증 실패: {e}", "confidence": 0.1, "estimated_mass_g": initial_estimation.get('estimated_mass_g', 0)}

    def _process_parsed_multimodal_response(self, parsed: dict, initial_estimation: dict) -> dict:
        foods = parsed.get('foods', [])
        food_verifications = []
        
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
                final_mass = food_info.get('verified_mass_g', 0)
                logging.info(f"시각적 추정 기반 질량: {final_mass}g")
            
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