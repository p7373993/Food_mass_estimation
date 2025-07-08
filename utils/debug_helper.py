"""
디버그 헬퍼 - 중간 과정의 상세한 디버그 정보를 출력
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any
import time
import os

class DebugHelper:
    """
    파이프라인의 각 단계별 디버그 정보를 상세하게 출력하는 클래스
    """
    
    def __init__(self, enable_debug: bool = True, simple_mode: bool = False):
        """디버그 헬퍼 초기화"""
        self.enable_debug = enable_debug
        self.simple_mode = simple_mode
        self.step_times = {}
        self.step_counter = 0
        
    def log_step_start(self, step_name: str):
        """단계 시작 로그"""
        if not self.enable_debug:
            return
            
        self.step_counter += 1
        print(f"\n{'='*60}")
        print(f"🔍 STEP {self.step_counter}: {step_name}")
        print(f"{'='*60}")
        self.step_times[step_name] = time.time()
        
    def log_step_end(self, step_name: str):
        """단계 종료 로그"""
        if not self.enable_debug:
            return
            
        if step_name in self.step_times:
            elapsed = time.time() - self.step_times[step_name]
            print(f"✅ {step_name} 완료 (소요시간: {elapsed:.2f}초)")
        
    def log_segmentation_debug(self, segmentation_results: Dict):
        """세그멘테이션 결과 디버그"""
        if not self.enable_debug:
            return
            
        if self.simple_mode:
            # 간단 모드: 핵심 정보만 출력
            food_objects = segmentation_results.get('food_objects', [])
            ref_objects = segmentation_results.get('reference_objects', [])
            print(f"📊 세그멘테이션 결과:")
            print(f"   - 음식 {len(food_objects)}개, 기준물체 {len(ref_objects)}개")
            for i, food in enumerate(food_objects):
                print(f"     {i+1}. {food.get('class_name', 'unknown')} (신뢰도: {food.get('confidence', 0):.3f})")
            for i, ref in enumerate(ref_objects):
                print(f"     기준: {ref.get('class_name', 'unknown')} (신뢰도: {ref.get('confidence', 0):.3f})")
            return
        
        # 기존 상세 모드
        print(f"\n📊 세그멘테이션 결과 상세:")
        print(f"   - 이미지 크기: {segmentation_results.get('image_shape', 'N/A')}")
        
        food_objects = segmentation_results.get('food_objects', [])
        print(f"   - 감지된 음식: {len(food_objects)}개")
        for i, food in enumerate(food_objects):
            print(f"     {i+1}. {food.get('class_name', 'unknown')} (신뢰도: {food.get('confidence', 0):.3f})")
            print(f"        위치: {food.get('position', {})}")
            print(f"        면적: {food.get('pixel_area', 0)} 픽셀")
            
        ref_objects = segmentation_results.get('reference_objects', [])
        print(f"   - 감지된 기준물체: {len(ref_objects)}개")
        for i, ref in enumerate(ref_objects):
            print(f"     {i+1}. {ref.get('class_name', 'unknown')} (신뢰도: {ref.get('confidence', 0):.3f})")
            print(f"        위치: {ref.get('position', {})}")
            print(f"        면적: {ref.get('pixel_area', 0)} 픽셀")
    
    def log_depth_debug(self, depth_results: Dict):
        """깊이 추정 결과 디버그"""
        if not self.enable_debug:
            return
            
        if self.simple_mode:
            # 간단 모드: 핵심 정보만 출력
            print(f"🌊 깊이 추정 결과 상세:")
            depth_stats = depth_results.get('depth_stats', {})
            print(f"   - 깊이 범위: {depth_stats.get('min_depth', 0):.3f} ~ {depth_stats.get('max_depth', 0):.3f}")
            print(f"   - 평균 깊이: {depth_stats.get('mean_depth', 0):.3f}")
            print(f"   - 깊이 표준편차: {depth_stats.get('std_depth', 0):.3f}")
            print(f"   - 깊이맵 크기: {depth_results.get('depth_map_shape', 'N/A')}")
            return
        
        # 기존 상세 모드
        print(f"\n🌊 깊이 추정 결과 상세:")
        depth_stats = depth_results.get('depth_stats', {})
        print(f"   - 깊이 범위: {depth_stats.get('min_depth', 0):.3f} ~ {depth_stats.get('max_depth', 0):.3f}")
        print(f"   - 평균 깊이: {depth_stats.get('mean_depth', 0):.3f}")
        print(f"   - 깊이 표준편차: {depth_stats.get('std_depth', 0):.3f}")
        print(f"   - 깊이맵 크기: {depth_results.get('depth_map_shape', 'N/A')}")
        
    def log_camera_debug(self, camera_info: Dict):
        """카메라 정보 디버그"""
        if not self.enable_debug:
            return
            
        print(f"\n📷 카메라 정보 상세:")
        if camera_info and camera_info.get('has_focal_length'):
            print(f"   - 초점거리: {camera_info.get('focal_length_mm', 0):.1f}mm")
            print(f"   - 35mm 환산: {camera_info.get('focal_length_35mm', 0):.0f}mm")
            print(f"   - 카메라 타입: {camera_info.get('camera_type', 'unknown')}")
            print(f"   - EXIF 정보: 사용 가능")
        else:
            print(f"   - EXIF 정보 없음 (기본값 사용)")
            if camera_info:
                print(f"   - 기본 초점거리: {camera_info.get('focal_length_mm', 0):.1f}mm")
                print(f"   - 기본 35mm 환산: {camera_info.get('focal_length_35mm', 0):.0f}mm")
                print(f"   - 카메라 타입: {camera_info.get('camera_type', 'unknown')}")
            else:
                print(f"   - 카메라 정보 없음")
    
    def log_depth_scale_debug(self, depth_scale_info: Dict):
        """깊이 스케일 디버그"""
        if not self.enable_debug:
            return
            
        if self.simple_mode:
            # 간단 모드: 핵심 정보만 출력
            print(f"📏 깊이 스케일 정보:")
            if depth_scale_info and depth_scale_info.get('has_scale'):
                print(f"   - 깊이 스케일: {depth_scale_info.get('depth_scale_cm_per_unit', 0):.6f} cm/unit")
                print(f"   - 계산 신뢰도: {depth_scale_info.get('confidence', 0):.3f}")
                print(f"   - 기준 물체 개수: {depth_scale_info.get('reference_count', 0)}개")
            else:
                print(f"   - 깊이 스케일 계산 실패")
            return
        
        # 기존 상세 모드
        print(f"\n📏 깊이 스케일 정보 상세:")
        if depth_scale_info and depth_scale_info.get('has_scale'):
            print(f"   - 깊이 스케일: {depth_scale_info.get('depth_scale_cm_per_unit', 0):.6f} cm/unit")
            print(f"   - 계산 신뢰도: {depth_scale_info.get('confidence', 0):.3f}")
            print(f"   - 기준 물체 개수: {depth_scale_info.get('reference_count', 0)}개")
            print(f"   - 계산 방법: {depth_scale_info.get('method', 'N/A')}")
            
            # 개별 스케일 정보
            individual_scales = depth_scale_info.get('individual_scales', [])
            for i, scale in enumerate(individual_scales):
                print(f"     {i+1}. {scale.get('object_name', 'unknown')}")
                print(f"        실제 높이: {scale.get('real_thickness_cm', 0):.1f}cm")
                print(f"        깊이 변화: {scale.get('depth_variation', 0):.3f}")
                print(f"        계산된 스케일: {scale.get('depth_scale_cm_per_unit', 0):.6f} cm/unit")
        else:
            print(f"   - 깊이 스케일 계산 실패")
            if depth_scale_info:
                print(f"   - 방법: {depth_scale_info.get('method', 'N/A')}")
            else:
                print(f"   - 정보 없음")

    def log_fallback_info_debug(self, fallback_info: Dict):
        """대안적 계산 정보 디버그"""
        if not self.enable_debug:
            return
            
        print(f"\n🔧 대안적 계산 정보 상세:")
        if not fallback_info:
            print(f"   - 정보 없음")
            return
            
        method = fallback_info.get('method', 'unknown')
        print(f"   - 계산 방법: {method}")
        print(f"   - 신뢰도: {fallback_info.get('confidence', 0):.3f}")
        print(f"   - 설명: {fallback_info.get('description', 'N/A')}")
        print(f"   - 추천 접근법: {fallback_info.get('recommended_approach', 'N/A')}")
        
        if method == 'camera_based':
            estimated_distance = fallback_info.get('estimated_distance_cm')
            pixel_scale = fallback_info.get('pixel_scale_cm_per_pixel')
            if estimated_distance:
                print(f"   - 추정 촬영 거리: {estimated_distance:.1f}cm")
            if pixel_scale:
                print(f"   - 픽셀 스케일: {pixel_scale:.6f} cm/pixel")
            
            calculation_notes = fallback_info.get('calculation_notes', [])
            if calculation_notes:
                print(f"   - 계산 참고사항:")
                for note in calculation_notes:
                    print(f"     • {note}")
                    
        elif method == 'empirical':
            assumptions = fallback_info.get('fallback_assumptions', [])
            if assumptions:
                print(f"   - 기본 가정:")
                for assumption in assumptions:
                    print(f"     • {assumption}")
            
            calculation_notes = fallback_info.get('calculation_notes', [])
            if calculation_notes:
                print(f"   - 주의사항:")
                for note in calculation_notes:
                    print(f"     • {note}")
    
    def log_features_debug(self, features: Dict):
        """특징 추출 결과 디버그"""
        if not self.enable_debug:
            return
            
        if self.simple_mode:
            # 간단 모드: 핵심 정보만 출력
            food_objects = features.get('food_objects', [])
            ref_objects = features.get('reference_objects', [])
            print(f"🎯 특징 추출 결과:")
            print(f"   - 음식 {len(food_objects)}개, 기준물체 {len(ref_objects)}개")
            for i, food in enumerate(food_objects):
                volume_info = food.get('real_volume_info', {})
                print(f"     {i+1}. {food.get('class_name', 'unknown')}: {volume_info.get('real_area_cm2', 0):.2f}cm² / {volume_info.get('real_volume_cm3', 0):.2f}cm³")
            for i, ref in enumerate(ref_objects):
                print(f"     기준: {ref.get('class_name', 'unknown')}: {ref.get('pixel_area', 0)} 픽셀")
            return
        
        # 기존 상세 모드
        print(f"\n🎯 특징 추출 결과 상세:")
        
        # 음식 특징
        food_objects = features.get('food_objects', [])
        print(f"   📍 음식 객체 특징 ({len(food_objects)}개):")
        for i, food in enumerate(food_objects):
            print(f"     {i+1}. {food.get('class_name', 'unknown')}")
            print(f"        픽셀 면적: {food.get('pixel_area', 0)}")
            
            # 깊이 정보
            depth_info = food.get('depth_info', {})
            print(f"        깊이 - 평균: {depth_info.get('mean_depth', 0):.3f}, 변화: {depth_info.get('depth_variation', 0):.3f}")
            
            # 실제 높이 정보 (새로 추가된 부분)
            if food.get('real_height_info'):
                height_info = food['real_height_info']
                print(f"        실제 높이: {height_info.get('real_height_cm', 0):.2f}cm (신뢰도: {height_info.get('confidence', 0):.2f})")
            
            # 실제 부피 정보 (새로 추가된 부분)
            if food.get('real_volume_info'):
                volume_info = food['real_volume_info']
                print(f"        실제 면적: {volume_info.get('real_area_cm2', 0):.2f}cm²")
                print(f"        실제 부피: {volume_info.get('real_volume_cm3', 0):.2f}cm³")
                print(f"        부피 계산: {'정확' if volume_info.get('calculation_accurate') else '근사'}")
            
            # 거리 정보 (카메라 정보 활용)
            if food.get('distance_info'):
                dist_info = food['distance_info']
                print(f"        추정 거리: {dist_info.get('estimated_distance_mm', 0):.1f}mm")
        
        # 기준물체 특징
        ref_objects = features.get('reference_objects', [])
        print(f"   🎯 기준물체 특징 ({len(ref_objects)}개):")
        for i, ref in enumerate(ref_objects):
            print(f"     {i+1}. {ref.get('class_name', 'unknown')}")
            print(f"        픽셀 면적: {ref.get('pixel_area', 0)}")
            depth_info = ref.get('depth_info', {})
            print(f"        깊이 - 평균: {depth_info.get('mean_depth', 0):.3f}, 변화: {depth_info.get('depth_variation', 0):.3f}")
            print(f"        실제 크기: {ref.get('real_size', {})}")
    
    def log_llm_prompt_debug(self, prompt: str):
        """LLM 프롬프트 디버그"""
        if not self.enable_debug:
            return
            
        print(f"\n🤖 LLM 프롬프트 상세:")
        print(f"{'='*50}")
        print(prompt)
        print(f"{'='*50}")
        print(f"프롬프트 길이: {len(prompt)} 문자")
    
    def log_multiple_foods_debug(self, individual_results: List[Dict], total_mass: float, overall_confidence: float):
        """여러 음식 개별 처리 결과를 디버그 출력합니다."""
        print(f"\n🍽️ 여러 음식 개별 처리 결과:")
        print(f"   - 총 음식 개수: {len(individual_results)}개")
        print(f"   - 총 질량: {total_mass:.1f}g")
        print(f"   - 전체 신뢰도: {overall_confidence:.3f}")
        
        for i, result in enumerate(individual_results):
            food_name = result.get("food_name", "unknown")
            mass = result.get("estimated_mass", 0)
            confidence = result.get("confidence", 0)
            print(f"   📍 음식 {i+1}: {food_name}")
            print(f"      - 질량: {mass:.1f}g")
            print(f"      - 신뢰도: {confidence:.3f}")
            print(f"      - 추정 근거: {result.get('reasoning', 'N/A')[:100]}...")
    
    def log_single_food_estimation_debug(self, features: Dict):
        """단일 음식 질량 추정 시작을 디버그 출력합니다."""
        food_objects = features.get("food_objects", [])
        if food_objects:
            food = food_objects[0]
            print(f"\n🔍 단일 음식 질량 추정:")
            print(f"   - 음식 종류: {food.get('class_name', 'unknown')}")
            print(f"   - 신뢰도: {food.get('confidence', 0):.3f}")
            print(f"   - 픽셀 면적: {food.get('pixel_area', 0):,}픽셀")
    
    def log_llm_response_debug(self, response: str, parsed_result: Dict = None):
        """LLM 응답을 디버그 출력합니다."""
        print(f"\n🎯 LLM 응답 상세:")
        print(f"==================================================")
        print(f"원본 응답:")
        print(f"```json")
        print(f"{response}")
        print(f"```")
        print(f"==================================================")
        
        if parsed_result:
            print(f"파싱된 결과:")
            print(f"   - 추정 질량: {parsed_result.get('mass', 0)}g")
            print(f"   - 신뢰도: {parsed_result.get('confidence', 0):.3f}")
            print(f"   - 전체 추정 근거: {parsed_result.get('reasoning', 'N/A')}")
        
        print(f"응답 길이: {len(response)} 문자")
    
    def log_summary_debug(self, result: Dict):
        """최종 결과 요약 디버그"""
        if not self.enable_debug:
            return
            
        if self.simple_mode:
            # 간단 모드: 핵심 정보만 출력
            initial_estimate = result.get("initial_estimate", {})
            final_estimate = result.get("final_estimate", {})
            
            print(f"💡 질량 추정 결과:")
            print(f"   - 초기 추정: {initial_estimate.get('estimated_mass', 0):.1f}g (신뢰도: {initial_estimate.get('confidence', 0):.3f})")
            print(f"   - 최종 결과: {final_estimate.get('final_mass', 0):.1f}g (신뢰도: {final_estimate.get('confidence', 0):.3f})")
            
            # 멀티모달 결과가 있는 경우
            if final_estimate.get('method') == 'multimodal_corrected':
                print(f"   - 멀티모달 결과: {final_estimate.get('multimodal_mass', 0):.1f}g")
                print(f"   - 보정 방법: {final_estimate.get('correction_method', 'N/A')}")
                print(f"   - 차이 비율: {final_estimate.get('difference_ratio', 0)*100:.1f}%")
            
            # 처리 시간
            processing_time = result.get("processing_time", 0)
            print(f"   - 처리 시간: {processing_time:.2f}초")
            return
        
        # 기존 상세 모드
        print(f"\n💡 최종 결과 요약:")
        
        # 처리 시간
        processing_time = result.get("processing_time", 0)
        print(f"   ⏱️  처리 시간: {processing_time:.2f}초")
        
        # 질량 추정 결과
        initial_estimate = result.get("initial_estimate", {})
        final_estimate = result.get("final_estimate", {})
        
        print(f"\n💡 질량 추정 결과:")
        
        # 여러 음식 개별 결과 표시
        individual_foods = initial_estimate.get("individual_foods", [])
        if individual_foods:
            print(f"   🍽️ 개별 음식 질량:")
            for i, food_result in enumerate(individual_foods):
                food_name = food_result.get("food_name", "unknown")
                mass = food_result.get("estimated_mass", 0)
                confidence = food_result.get("confidence", 0)
                print(f"      {i+1}. {food_name}: {mass:.1f}g (신뢰도: {confidence:.3f})")
        
        # 초기 추정값
        initial_mass = initial_estimate.get("estimated_mass", 0)
        initial_confidence = initial_estimate.get("confidence", 0)
        print(f"   📊 초기 추정값: {initial_mass:.1f}g (신뢰도: {initial_confidence:.3f})")
        
        # 멀티모달 결과가 있는 경우
        if final_estimate.get("method") == "multimodal_corrected":
            multimodal_mass = final_estimate.get("multimodal_mass", 0)
            multimodal_confidence = final_estimate.get("multimodal_confidence", 0)
            print(f"   🔄 멀티모달 결과: {multimodal_mass:.1f}g (신뢰도: {multimodal_confidence:.3f})")
        
        # 최종 결과  
        final_mass = final_estimate.get("final_mass", 0)
        final_confidence = final_estimate.get("confidence", 0)
        print(f"   ✅ 최종 질량: {final_mass:.1f}g (신뢰도: {final_confidence:.3f})")
        
        # 추정 근거
        reasoning = final_estimate.get("reasoning", "N/A")
        print(f"   🔍 추정 근거: {reasoning}")
        
        # 보조 결과
        if final_estimate.get("method") == "multimodal_corrected":
            print(f"\n🔍 보조 결과:")
            print(f"   📈 초기 추정값: {initial_mass:.1f}g (신뢰도: {initial_confidence:.3f})")
            print(f"   🔄 멀티모달 결과: {multimodal_mass:.1f}g (신뢰도: {multimodal_confidence:.3f})")
            
            # 보정 정보
            correction_method = final_estimate.get("correction_method", "N/A")
            correction_reason = final_estimate.get("correction_reason", "N/A") 
            difference_ratio = final_estimate.get("difference_ratio", 0) * 100
            print(f"   ⚖️ 보정 정보: 차이 비율 {difference_ratio:.1f}%, {correction_method} 방법")
            print(f"   💬 보정 근거: {correction_reason}")
        
        # 전체 처리 통계
        segmentation_results = result.get("segmentation_results", {})
        food_count = len(segmentation_results.get("food_objects", []))
        ref_count = len(segmentation_results.get("reference_objects", []))
        
        print(f"\n📊 처리 통계:")
        print(f"   🍽️  감지된 음식: {food_count}개")
        print(f"   📏 기준 물체: {ref_count}개")
        print(f"   🔧 처리 방법: {final_estimate.get('method', 'N/A')}")
        print(f"   ⏱️  처리 시간: {processing_time:.2f}초")
        
        # 분석 결과
        reference_analysis = result.get("reference_analysis", {})
        if reference_analysis:
            has_reference = reference_analysis.get("has_reference", False)
            ref_confidence = reference_analysis.get("confidence", 0)
            print(f"   📈 기준 물체 분석: {'사용' if has_reference else '미사용'}")
            if has_reference:
                print(f"   🎯 기준 물체 신뢰도: {ref_confidence:.3f}")
        
        # 추가 정보
        if final_estimate.get("adjustment"):
            adjustment = final_estimate.get("adjustment")
            print(f"   🔄 LLM 조정: {adjustment}")
        
        # 제품 정보 (공산품인 경우)
        if final_estimate.get("product_info") and final_estimate.get("product_info") != "일반음식":
            product_info = final_estimate.get("product_info")
            print(f"   🏪 제품 정보: {product_info}")
        
        print(f"\n{'='*60}")
        print(f"🎯 최종 결과: {final_mass:.1f}g (신뢰도: {final_confidence:.3f})")
        print(f"{'='*60}")
        
    def print_separator(self, title: str = ""):
        """구분선 출력"""
        if not self.enable_debug:
            return
        print(f"\n{'='*60}")
        if title:
            print(f"{title}")
            print(f"{'='*60}") 