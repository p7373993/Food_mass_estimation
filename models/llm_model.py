import logging
import google.generativeai as genai
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
    
    def estimate_mass_from_features(self, features: dict) -> dict:
        """
        추출된 특징을 바탕으로 LLM에게 질량 추정을 요청합니다.

        Args:
            features (dict): 음식 객체, 기준 객체, 깊이 정보 등을 담은 딕셔너리.

        Returns:
            dict: 추정된 질량, 신뢰도, 설명 등을 담은 딕셔너리.
        """
        if self._model is None:
            return {"error": "LLM 모델이 초기화되지 않았습니다."}

        prompt = self._build_prompt(features)
        
        try:
            response = self._model.generate_content(prompt)
            # 여기서는 응답을 파싱하는 간단한 예시를 보여줍니다.
            # 실제로는 더 정교한 JSON 파싱 및 오류 처리가 필요합니다.
            mass_info = self._parse_response(response.text)
            return mass_info

        except Exception as e:
            logging.error(f"LLM 질량 추정 중 오류 발생: {e}")
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


# 싱글톤 인스턴스 생성
llm_estimator = LLMMassEstimator() 