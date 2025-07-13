import logging
import numpy as np
import google.generativeai as genai
from typing import Dict

from config.settings import settings
from utils.base_model import BaseModel

class LLMMassEstimator(BaseModel):
    """
    LLM (Gemini)을 사용하여 질량을 추정하는 래퍼 클래스.
    - BaseModel을 상속받아 싱글톤 패턴과 공통 로직 사용
    - 중앙 설정 파일(settings)을 사용합니다.
    """
    
    def get_model_name(self) -> str:
        return "Gemini LLM 모델"
    
    def _initialize_model(self) -> None:
        """Gemini 모델 초기화"""
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
        """
        추출된 특징을 바탕으로 LLM에게 질량 추정을 요청합니다.

        Args:
            features (dict): 음식 객체, 기준 객체, 깊이 정보 등을 담은 딕셔너리.
            debug_helper: 디버그 헬퍼 인스턴스 (선택적)

        Returns:
            dict: 추정된 질량, 신뢰도, 설명 등을 담은 딕셔너리.
        """
        if self._model is None:
            return {"error": "LLM 모델이 초기화되지 않았습니다."}

        prompt = self._build_prompt(features)
        
        try:
            response = self._model.generate_content(
                prompt, 
                generation_config=genai.types.GenerationConfig(
                    temperature=settings.LLM_TEMPERATURE,
                    top_p=settings.LLM_TOP_P
                )
            )
            
            # 응답 파싱
            mass_info = self._parse_response(response.text)
            
            # 디버그 정보 출력
            if debug_helper:
                debug_helper.log_initial_mass_calculation_debug(
                    features, prompt, response.text, mass_info
                )
            
            return mass_info

        except Exception as e:
            logging.error(f"LLM 질량 추정 중 오류 발생: {e}")
            return {"error": str(e)}

    def verify_mass_with_multimodal(self, image: np.ndarray, initial_estimation: dict, features: dict) -> dict:
        """
        원본 이미지와 초기 추정 결과를 바탕으로 멀티모달 검증을 수행합니다.

        Args:
            image (np.ndarray): 원본 이미지
            initial_estimation (dict): 1차 LLM 추정 결과
            features (dict): 추출된 특징 정보

        Returns:
            dict: 검증된 질량 추정 결과
        """
        if self._model is None:
            return {"error": "LLM 모델이 초기화되지 않았습니다."}

        try:
            # 멀티모달 모델 선택 (별도 모델이 설정되어 있으면 사용, 없으면 기본 모델 사용)
            multimodal_model_name = settings.MULTIMODAL_MODEL_NAME or settings.LLM_MODEL_NAME
            if multimodal_model_name != settings.LLM_MODEL_NAME:
                # 별도 멀티모달 모델 사용
                multimodal_model = genai.GenerativeModel(multimodal_model_name)
            else:
                # 기본 모델 사용
                multimodal_model = self._model
            
            # 이미지를 base64로 인코딩
            import base64
            import cv2
            
            # 이미지를 RGB로 변환 (Gemini는 RGB 형식 필요)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 이미지 크기 조정 (API 제한 고려)
            max_size = 1024  # 1536에서 1024로 낮춰서 이미지 품질 저하 방지
            h, w = image_rgb.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_rgb = cv2.resize(image_rgb, (new_w, new_h))
            
            # JPEG로 인코딩
            _, buffer = cv2.imencode('.jpg', image_rgb)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 멀티모달 프롬프트 생성
            prompt = self._build_multimodal_prompt(initial_estimation, features)
            
            # 멀티모달 요청 생성
            multimodal_content = [
                {
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
            
            # 멀티모달 모델로 요청
            response = multimodal_model.generate_content(
                multimodal_content, 
                generation_config=genai.types.GenerationConfig(
                    temperature=settings.LLM_TEMPERATURE,
                    top_p=settings.LLM_TOP_P
                )
            )
            verification_result = self._parse_multimodal_response(response.text, initial_estimation)
            
            return verification_result
            
        except Exception as e:
            logging.error(f"멀티모달 검증 중 오류 발생: {e}")
            return {"error": str(e)}

    def _build_prompt(self, features: dict) -> str:
        """
        사용자 제공 코드를 기반으로, LLM이 직접 계산할 수 있도록 개선된 질량 추정 프롬프트를 생성합니다.
        """
        food_objects = features.get("food_objects", [])
        reference_objects = features.get("reference_objects", [])
        depth_scale_info = features.get("depth_scale_info", {})
        
        if not food_objects:
            return '음식이 감지되지 않았습니다. {"mass": 0, "confidence": 0.1, "reasoning": "음식 감지 실패"}'
        
        food = food_objects[0]
        has_reference = len(reference_objects) > 0
        has_depth_scale = depth_scale_info.get('has_scale', False)
        
        prompt = "음식 질량 추정 분석:\n\n"
        prompt += f"🍽️ 음식 정보:\n"
        prompt += f"  - 종류: {food.get('class_name', '알수없음')}\n"
        prompt += f"  - 픽셀 면적: {food.get('pixel_area', 0):,}픽셀\n"
        
        food_depth = food.get('depth_info', {})
        prompt += f"  - 평균 깊이값(상대적): {food_depth.get('mean_depth', 0.0):.3f}\n"
        prompt += f"  - 깊이 변화량(상대적): {food_depth.get('depth_variation', 0.0):.3f}\n"
        
        if has_reference:
            ref_obj = reference_objects[0]
            prompt += f"\n📏 기준 물체 정보:\n"
            prompt += f"  - 종류: {ref_obj.get('class_name')}\n"
            real_size = ref_obj.get('real_size', {})
            if real_size:
                prompt += f"  - 실제 크기: {real_size.get('width', 0):.1f}cm × {real_size.get('height', 0):.1f}cm, 두께: {real_size.get('thickness', 0):.1f}cm\n"
        
        if has_depth_scale:
            prompt += f"\n🔍 계산된 실제 스케일:\n"
            prompt += f"  - 깊이 스케일: {depth_scale_info.get('depth_scale_cm_per_unit', 0.0):.4f} cm/unit (상대적 깊이 1단위당 실제 cm)\n"
            if depth_scale_info.get('pixel_per_cm2_ratio'):
                prompt += f"  - 면적 비율: {depth_scale_info.get('pixel_per_cm2_ratio'):.2f} pixels/cm² (1 제곱센티미터당 픽셀 수)\n"

        prompt += f"\n🎯 계산 과제:\n"
        prompt += f"위 정보를 바탕으로 음식('{food.get('class_name', '알수없음')}')의 질량을 g(그램) 단위로 추정하세요. 부피(cm³)를 먼저 계산한 후, 일반적인 음식 밀도(약 0.8~1.2 g/cm³)를 적용하세요.\n"
        
        prompt += f"\n💡 계산 가이드:\n"
        if has_reference and has_depth_scale and depth_scale_info.get('pixel_per_cm2_ratio'):
            prompt += f"1. '픽셀 면적'을 '면적 비율'로 나누어 음식의 실제 면적(cm²)을 계산하세요.\n"
            prompt += f"2. '깊이 변화량'과 '깊이 스케일'을 곱하여 음식의 실제 높이(cm)를 추정하세요.\n"
            prompt += f"3. 추정된 실제 면적과 높이를 곱해 부피(cm³)를 계산하세요. (형태 보정 계수 0.6 적용)\n"
            prompt += f"4. 계산된 부피에 음식의 예상 밀도(g/cm³)를 곱해 최종 질량(g)을 계산하세요.\n"
        else:
            prompt += f"정확한 스케일 정보가 없으므로, 음식의 픽셀 면적과 일반적인 음식 크기를 고려하여 경험적으로 추정하세요. 신뢰도를 낮게 설정하세요.\n"

        prompt += f"\n📋 응답 형식 (JSON):\n"
        prompt += f'{{"estimated_mass_g": <추정 질량(g)>, "confidence": <신뢰도(0.0~1.0)>, "reasoning": "<계산 과정 및 근거 요약>"}}'
        
        return prompt.strip()

    def _parse_response(self, response_text: str) -> dict:
        """LLM의 응답 텍스트를 파싱하여 딕셔너리로 변환합니다."""
        try:
            # LLM 응답에서 JSON 부분만 추출
            json_part = response_text.split('```json')[-1].split('```')[0].strip()
            import json
            return json.loads(json_part)
        except Exception as e:
            logging.error(f"LLM 응답 파싱 실패: {e}\n원본 응답: {response_text}")
            return {"error": "LLM 응답을 파싱할 수 없습니다."}

    def _build_multimodal_prompt(self, initial_estimation: dict, features: dict) -> str:
        """
        멀티모달 검증을 위한 프롬프트를 생성합니다.
        """
        initial_mass = initial_estimation.get('estimated_mass_g', 0)
        initial_confidence = initial_estimation.get('confidence', 0)
        initial_reasoning = initial_estimation.get('reasoning', '')
        
        food_objects = features.get("food_objects", [])
        reference_objects = features.get("reference_objects", [])
        
        prompt = f"""
음식 질량 추정 멀티모달 검증:

📊 초기 추정 결과:
- 추정 질량: {initial_mass:.1f}g
- 신뢰도: {initial_confidence:.3f}
- 추정 근거: {initial_reasoning}

🍽️ 감지된 음식 정보:
"""
        
        for i, food in enumerate(food_objects):
            prompt += f"- 음식 {i+1}: {food.get('class_name', '알수없음')}\n"
            prompt += f"  픽셀 면적: {food.get('pixel_area', 0):,}픽셀\n"
            
            depth_info = food.get('depth_info', {})
            prompt += f"  깊이 변화량: {depth_info.get('depth_variation', 0):.3f}\n"
        
        if reference_objects:
            prompt += f"\n📏 기준 물체:\n"
            for ref in reference_objects:
                prompt += f"- {ref.get('class_name', '알수없음')}\n"
        
        prompt += f"""
🔍 검증 과제:
위 이미지를 직접 보고 다음을 분석하세요:

1. **음식 종류 식별**: 
   - 공산품(패키지된 음식)인가요?
   - 자연식품(과일, 채소, 생선 등)인가요?
   - 조리된 음식(밥, 국, 반찬 등)인가요?

2. **음식 이름/종류 판단** (모든 경우):
   - 정확히 어떤 음식인지 식별할 수 있나요?
   - 구체적인 음식 이름이나 종류를 알려주세요 (예: "김치찌개", "사과", "삼겹살", "팔도 비빔면" 등)

3. **라벨 정보 확인** (공산품인 경우):
   - 패키지에 무게 정보가 보이나요?
   - 제품명과 무게가 표시되어 있나요?
   - 라벨 정보를 읽을 수 있나요?
   
   **📝 라벨 인식 지침 (완화된 버전):**
   - 라벨의 텍스트 부분을 주의 깊게 살펴보세요
   - 제품명, 브랜드명, 무게 정보를 읽어주세요
   - **부분적으로 보이거나 흐려도 추정 가능한 정보라도 제공해주세요**
   - **일부라도 읽히는 경우 추정 가능한 정보를 알려주세요**
   - 라벨이 완전히 불분명한 경우에만 "라벨 텍스트 불분명"이라고 표시하세요
   - **중요: 가능한 한 많은 정보를 추출하려고 노력하세요**

4. **공산품 식별** (라벨이 불분명한 경우):
   - 어떤 종류의 공산품인지 판단할 수 있나요? (예: 컵라면, 과자, 음료, 통조림 등)
   - 제품의 브랜드나 제품명을 추측할 수 있나요?
   - 해당 제품의 일반적인 무게는 얼마인가요?

5. **질량 추정 검증**:
   - 초기 추정({initial_mass:.1f}g)이 합리적인가요?
   - 시각적으로 보이는 양과 일치하나요?
   - 기준 물체와의 상대적 크기는 적절한가요?

📋 응답 형식 (JSON):
{{
    "food_type": "<packaged_product|natural_food|cooked_food>",
    "food_name": "<구체적인 음식 이름 (예: 김치찌개, 사과, 팔도 비빔면 등)>",
    "has_clear_label": <true/false>,
    "label_mass_g": <라벨에 표시된 무게, 없으면 null>,
    "product_name": "<제품명, 없으면 null>",
    "product_category": "<제품 카테고리 (컵라면, 과자, 음료 등), 없으면 null>",
    "estimated_product_mass_g": <해당 제품의 일반적인 무게 추정, 없으면 null>,
    "verified_mass_g": <검증된 질량(g)>,
    "confidence": <신뢰도(0.0~1.0)>,
    "verification_method": "<label_based|product_estimated|multimodal_visual|hybrid>",
    "reasoning": "<검증 근거>",
    "adjustment_needed": <true/false>,
    "adjustment_reason": "<조정 이유>"
}}

**중요**: 
- 모든 경우에 food_name을 반드시 포함하세요 (음식의 구체적인 이름/종류)
- **라벨 텍스트를 최대한 읽으려고 노력하되, 부분적 정보라도 활용하세요**
- 공산품이고 라벨 정보가 명확하면 label_mass_g를 최종 결과로 사용
- 공산품이지만 라벨이 불분명하면 제품 종류를 판단하고 estimated_product_mass_g 사용
- 자연식품이나 조리된 음식이면 초기 추정에 더 가중치를 둠
"""
        
        return prompt.strip()

    def _parse_multimodal_response(self, response_text: str, initial_estimation: dict) -> dict:
        """
        멀티모달 응답을 파싱하고 음식 종류에 따라 다른 전략을 적용합니다.
        개선된 fallback 파싱 기능 포함.
        """
        try:
            import json
            import re
            
            # 1차: 완전한 JSON 추출 시도
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return self._process_parsed_response(parsed, initial_estimation)
                except json.JSONDecodeError:
                    logging.warning("JSON 파싱 실패, fallback 파싱 시도")
            
            # 2차: fallback 파싱 - 키워드 기반 추출
            fallback_result = self._fallback_parse_response(response_text, initial_estimation)
            if fallback_result:
                return fallback_result
            
            # 3차: 완전 실패 시 초기 추정 결과 반환
            return {
                "estimated_mass_g": initial_estimation.get('estimated_mass_g', 0),
                "confidence": initial_estimation.get('confidence', 0) * 0.8,
                "reasoning": "멀티모달 검증 실패, 초기 추정 결과 사용",
                "verification_method": "fallback_to_initial",
                "initial_estimation": initial_estimation
            }
                
        except Exception as e:
            logging.error(f"멀티모달 응답 파싱 오류: {e}")
            return {
                "estimated_mass_g": initial_estimation.get('estimated_mass_g', 0),
                "confidence": initial_estimation.get('confidence', 0) * 0.8,
                "reasoning": f"멀티모달 검증 오류: {e}",
                "verification_method": "error_fallback",
                "initial_estimation": initial_estimation
            }

    def _fallback_parse_response(self, response_text: str, initial_estimation: dict) -> dict:
        """
        키워드 기반 fallback 파싱을 수행합니다.
        """
        try:
            # 기본값 설정
            food_type = 'unknown'
            food_name = '알수없음'
            has_clear_label = False
            label_mass = None
            product_name = None
            product_category = None
            estimated_product_mass = None
            verified_mass = initial_estimation.get('estimated_mass_g', 0)
            confidence = 0.5
            reasoning = "fallback 파싱으로 추출"
            
            # 키워드 기반 추출
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip().lower()
                
                # 음식 종류 추출
                if 'packaged_product' in line or '공산품' in line or '패키지' in line:
                    food_type = 'packaged_product'
                elif 'natural_food' in line or '자연식품' in line or '과일' in line or '채소' in line:
                    food_type = 'natural_food'
                elif 'cooked_food' in line or '조리된' in line or '밥' in line or '국' in line:
                    food_type = 'cooked_food'
                
                # 음식 이름 추출
                if 'food_name' in line and ':' in line:
                    name_part = line.split(':', 1)[1].strip().strip('"\'')
                    if name_part and name_part not in ['null', 'none', '알수없음']:
                        food_name = name_part
                
                # 라벨 정보 추출
                if 'has_clear_label' in line:
                    if 'true' in line or 'true' in line:
                        has_clear_label = True
                
                # 라벨 무게 추출
                if 'label_mass_g' in line and ':' in line:
                    mass_part = line.split(':', 1)[1].strip()
                    try:
                        # 숫자만 추출
                        mass_match = re.search(r'(\d+(?:\.\d+)?)', mass_part)
                        if mass_match:
                            label_mass = float(mass_match.group(1))
                    except:
                        pass
                
                # 제품명 추출
                if 'product_name' in line and ':' in line:
                    name_part = line.split(':', 1)[1].strip().strip('"\'')
                    if name_part and name_part not in ['null', 'none']:
                        product_name = name_part
                
                # 검증된 무게 추출
                if 'verified_mass_g' in line and ':' in line:
                    mass_part = line.split(':', 1)[1].strip()
                    try:
                        mass_match = re.search(r'(\d+(?:\.\d+)?)', mass_part)
                        if mass_match:
                            verified_mass = float(mass_match.group(1))
                    except:
                        pass
                
                # 신뢰도 추출
                if 'confidence' in line and ':' in line:
                    conf_part = line.split(':', 1)[1].strip()
                    try:
                        conf_match = re.search(r'(\d+(?:\.\d+)?)', conf_part)
                        if conf_match:
                            confidence = float(conf_match.group(1))
                    except:
                        pass
            
            # 추출된 정보로 결과 생성
            return self._process_parsed_response({
                'food_type': food_type,
                'food_name': food_name,
                'has_clear_label': has_clear_label,
                'label_mass_g': label_mass,
                'product_name': product_name,
                'product_category': product_category,
                'estimated_product_mass_g': estimated_product_mass,
                'verified_mass_g': verified_mass,
                'confidence': confidence,
                'reasoning': reasoning
            }, initial_estimation)
            
        except Exception as e:
            logging.error(f"Fallback 파싱 오류: {e}")
            return None

    def _process_parsed_response(self, parsed: dict, initial_estimation: dict) -> dict:
        """
        파싱된 응답을 처리하여 최종 결과를 생성합니다.
        """
        food_type = parsed.get('food_type', 'unknown')
        food_name = parsed.get('food_name', '알수없음')
        has_clear_label = parsed.get('has_clear_label', False)
        label_mass = parsed.get('label_mass_g')
        product_name = parsed.get('product_name')
        product_category = parsed.get('product_category')
        estimated_product_mass = parsed.get('estimated_product_mass_g')
        verified_mass = parsed.get('verified_mass_g', initial_estimation.get('estimated_mass_g', 0))
        confidence = parsed.get('confidence', initial_estimation.get('confidence', 0))
        reasoning = parsed.get('reasoning', '멀티모달 검증 완료')
        
        initial_mass = initial_estimation.get('estimated_mass_g', 0)
        initial_confidence = initial_estimation.get('confidence', 0)
        
        # 라벨 텍스트가 불분명하거나 추측적인 경우 처리
        if food_name in ['라벨 텍스트 불분명', '알수없음', 'unknown'] or '추측' in food_name.lower():
            has_clear_label = False
            food_name = '라벨 텍스트 불분명'
        
        # 음식 종류와 라벨 정보에 따른 전략 적용
        if food_type == 'packaged_product' and has_clear_label and label_mass:
            # 공산품 + 정확한 라벨 정보 → 라벨 정보 사용
            final_mass = label_mass
            final_confidence = 0.95  # 라벨 정보는 매우 신뢰할 수 있음
            verification_method = "label_based"
            reasoning = f"공산품 라벨 정보 사용: {food_name} - {label_mass}g"
            
        elif food_type == 'packaged_product' and not has_clear_label:
            # 공산품 + 라벨 정보 없음 → 초기 추정값에 더 가중치를 두고, 제품 카테고리 정보는 보조로 사용
            if product_category and estimated_product_mass is not None:
                # 초기 추정값에 더 가중치 (70%), 제품 카테고리 추정값은 보조 (30%)
                final_mass = (initial_mass * 0.7) + (estimated_product_mass * 0.3)
                final_confidence = (initial_confidence * 0.6) + (confidence * 0.4)
                verification_method = "hybrid_packaged_with_category"
                reasoning = f"공산품이지만 라벨 불분명: {food_name} ({product_category}) - 초기 추정({initial_mass:.1f}g)에 가중치, 제품 카테고리 추정({estimated_product_mass:.1f}g) 보조"
            else:
                # 제품 카테고리가 없으면 초기 모델과 멀티모달 혼합
                final_mass = (initial_mass * 0.6) + (verified_mass * 0.4)
                final_confidence = (initial_confidence * 0.5) + (confidence * 0.5)
                verification_method = "hybrid_packaged"
                reasoning = f"공산품이지만 라벨 불분명: {food_name} - 초기 추정({initial_mass:.1f}g)에 가중치, 시각적 검증({verified_mass:.1f}g) 보조"
        
        else:
            # 자연식품/조리된 음식 → 초기 모델에 더 가중치
            final_mass = (initial_mass * 0.7) + (verified_mass * 0.3)
            final_confidence = (initial_confidence * 0.6) + (confidence * 0.4)
            verification_method = "hybrid_natural"
            reasoning = f"{food_name} ({food_type}): 초기 추정({initial_mass:.1f}g)에 더 가중치, 시각적 검증({verified_mass:.1f}g) 보조"
        
        return {
            "estimated_mass_g": final_mass,
            "confidence": final_confidence,
            "reasoning": reasoning,
            "verification_method": verification_method,
            "initial_estimation": initial_estimation,
            "food_type": food_type,
            "food_name": food_name,
            "has_clear_label": has_clear_label,
            "label_mass_g": label_mass,
            "product_name": product_name,
            "product_category": product_category,
            "estimated_product_mass_g": estimated_product_mass,
            "multimodal_mass_g": verified_mass,
            "adjustment_needed": abs(final_mass - initial_mass) > 10,  # 10g 이상 차이나면 조정 필요
            "adjustment_reason": parsed.get('adjustment_reason', '')
        }


# 싱글톤 인스턴스 생성
llm_estimator = LLMMassEstimator() 