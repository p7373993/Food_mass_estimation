#!/usr/bin/env python3
"""
AI 기반 음식 질량 추정 시스템
YOLO-Segment + MiDaS + LLM 파이프라인

사용법:
    python main.py <이미지_경로> [옵션]
    
예시:
    python main.py test_image.jpg
    python main.py test_image.jpg --debug --no-multimodal
"""

import argparse
import sys
import os
import json
from typing import Dict, List
import logging

# 현재 디렉토리를 패스에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.estimation_service import mass_estimation_service
from config.settings import settings
from utils.logging_utils import setup_logging


def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="AI 기반 음식 질량 추정 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py image.jpg                    # 기본 설정으로 실행
  python main.py image.jpg --debug            # 디버그 모드로 실행
  python main.py image.jpg --no-multimodal    # 멀티모달 검증 비활성화
  python main.py --config                     # 설정 확인
        """
    )
    
    # 위치 인수
    parser.add_argument(
        'image_path',
        nargs='*',
        help='처리할 이미지 파일 경로'
    )
    
    # 옵션 인수
    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 모드 활성화'
    )
    
    parser.add_argument(
        '--simple-debug',
        action='store_true',
        help='간단 디버그 모드 활성화 (핵심 정보만 출력)'
    )
    
    parser.add_argument(
        '--no-multimodal',
        action='store_true',
        help='멀티모달 검증 비활성화'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='현재 설정 출력'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='결과 저장 경로'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API 키'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='LLM 모델 이름 (기본값: gpt-4)'
    )
    
    return parser.parse_args()


def print_config():
    """현재 설정 출력"""
    print("=== 현재 설정 ===")
    
    # settings 객체의 모든 속성을 출력
    for key in dir(settings):
        if not key.startswith('_') and not callable(getattr(settings, key)):
            value = getattr(settings, key)
            if 'API_KEY' in key and value:
                # API 키는 일부만 표시
                masked_value = f"{value[:8]}..." if len(str(value)) > 8 else "***"
                print(f"{key}: {masked_value}")
            else:
                print(f"{key}: {value}")
    
    print("\n=== 설정 검증 ===")
    # 기본적인 검증 로직
    errors = []
    if not settings.GEMINI_API_KEY and not settings.OPENAI_API_KEY:
        errors.append("API 키가 설정되지 않았습니다.")
    
    if errors:
        print("오류:")
        for error in errors:
            print(f"  {error}")
    else:
        print("모든 설정이 유효합니다.")



def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 설정 확인
    if args.config:
        print_config()
        return
    
    # 이미지 경로 확인
    if not args.image_path:
        print("오류: 이미지 경로를 입력해주세요.")
        print("사용법: python main.py <이미지_경로>")
        return
    
    # 설정 업데이트 (런타임에 설정 변경)
    if args.api_key:
        if settings.LLM_PROVIDER == "gemini":
            settings.GEMINI_API_KEY = args.api_key
        else:
            settings.OPENAI_API_KEY = args.api_key
    
    if args.model:
        settings.LLM_MODEL_NAME = args.model
    
    debug_mode = args.debug or args.simple_debug
    simple_debug = args.simple_debug
    
    # 로깅 설정
    setup_logging(debug_mode, simple_debug)
    logger = logging.getLogger(__name__)
    
    # 설정 검증
    if not settings.GEMINI_API_KEY and not settings.OPENAI_API_KEY:
        print("설정 오류:")
        print("  API 키가 설정되지 않았습니다. .env 파일을 확인하거나 --api-key 옵션을 사용하세요.")
        return
    
    try:
        # 이미지 처리
        if args.image_path:
            # 단일 이미지 처리
            image_path = args.image_path[0]
            logger.info(f"단일 이미지 처리: {image_path}")
            
            # 입력 검증
            if not os.path.exists(image_path):
                print(f"입력 오류: 파일이 존재하지 않습니다: {image_path}")
                return
            
            # 이미지 로드
            from models.yolo_model import load_image
            image = load_image(image_path)
            if image is None:
                print(f"입력 오류: 이미지를 로드할 수 없습니다: {image_path}")
                return
            
            # 질량 추정
            result = mass_estimation_service.run_pipeline(image, image_path)
            
            # 결과 출력
            if "error" in result:
                print(f"오류: {result['error']}")
                return
            
            # 간단한 결과 출력
            print(f"\n=== 질량 추정 결과 ===")
            print(f"이미지: {image_path}")
            
            mass_estimation = result.get('mass_estimation', {})
            if 'error' in mass_estimation:
                print(f"LLM 추정 오류: {mass_estimation['error']}")
            else:
                estimated_mass = mass_estimation.get('estimated_mass_g', 'N/A')
                confidence = mass_estimation.get('confidence', 'N/A')
                reasoning = mass_estimation.get('reasoning', 'N/A')
                
                print(f"추정 질량: {estimated_mass}g")
                print(f"신뢰도: {confidence}")
                print(f"추정 근거: {reasoning}")
            
            # 결과 파일 저장
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                print(f"\n결과 파일 저장: {args.output}")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        print(f"오류: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 