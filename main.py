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

from pipeline.mass_estimation_pipeline import MassEstimationPipeline
from pipeline.config import Config


def setup_logging(debug: bool = False, simple_debug: bool = False):
    """로깅 설정"""
    if simple_debug:
        # 간단 디버그 모드: WARNING 이상만 출력
        level = logging.WARNING
        
        # 외부 라이브러리 로그 비활성화
        logging.getLogger('PIL').setLevel(logging.ERROR)
        logging.getLogger('PIL.TiffImagePlugin').setLevel(logging.ERROR)
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        logging.getLogger('ultralytics.yolo.utils').setLevel(logging.ERROR)
        logging.getLogger('ultralytics.yolo.v8').setLevel(logging.ERROR)
        logging.getLogger('torch').setLevel(logging.ERROR)
        logging.getLogger('torch.hub').setLevel(logging.ERROR)
        logging.getLogger('timm').setLevel(logging.ERROR)
        logging.getLogger('timm.models').setLevel(logging.ERROR)
        logging.getLogger('transformers').setLevel(logging.ERROR)
        
        # 주요 모듈들도 INFO 이상으로 설정
        logging.getLogger('root').setLevel(logging.INFO)
        logging.getLogger('pipeline.mass_estimation_pipeline').setLevel(logging.INFO)
        logging.getLogger('__main__').setLevel(logging.INFO)
        
        # Python warnings 비활성화
        import warnings
        warnings.filterwarnings('ignore')
        
    elif debug:
        # 일반 디버그 모드: 모든 정보 출력
        level = logging.DEBUG
    else:
        # 기본 모드: INFO 이상만 출력
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(os.path.join(Config.LOGS_DIR, 'main.log')),
            logging.StreamHandler()
        ]
    )


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
    settings = Config.get_all_settings()
    
    for key, value in settings.items():
        if 'API_KEY' in key and value:
            # API 키는 일부만 표시
            masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
            print(f"{key}: {masked_value}")
        else:
            print(f"{key}: {value}")
    
    print("\n=== 설정 검증 ===")
    errors = Config.validate_settings()
    if errors:
        print("오류:")
        for key, error in errors.items():
            print(f"  {key}: {error}")
    else:
        print("모든 설정이 유효합니다.")


def print_result_summary(result: Dict):
    """결과 요약 출력"""
    print("\n" + "="*60)
    print("질량 추정 결과")
    print("="*60)
    
    # 기본 정보
    print(f"이미지: {result['image_path']}")
    print(f"처리 시간: {result['processing_time']:.2f}초")
    
    # 감지된 객체
    food_objects = result['segmentation_results']['food_objects']
    ref_objects = result['segmentation_results']['reference_objects']
    
    print(f"\n감지된 음식: {len(food_objects)}개")
    for i, food in enumerate(food_objects):
        print(f"  {i+1}. {food['class_name']} (신뢰도: {food['confidence']:.2f})")
    
    print(f"\n감지된 기준 물체: {len(ref_objects)}개")
    for i, ref in enumerate(ref_objects):
        print(f"  {i+1}. {ref['class_name']} (신뢰도: {ref['confidence']:.2f})")
    
    # 질량 추정 결과
    initial_estimate = result['initial_estimate']
    final_estimate = result['final_estimate']
    
    print(f"\n초기 추정 질량: {initial_estimate['estimated_mass']:.1f}g")
    print(f"초기 추정 신뢰도: {initial_estimate['confidence']:.2f}")
    
    print(f"\n최종 추정 질량: {final_estimate['final_mass']:.1f}g")
    print(f"최종 추정 신뢰도: {final_estimate['confidence']:.2f}")
    print(f"추정 방법: {final_estimate['method']}")
    
    # 추정 근거
    print(f"\n추정 근거:")
    print(f"  {final_estimate['reasoning']}")
    
    # 기준 물체 분석
    ref_analysis = result['reference_analysis']
    if ref_analysis['has_reference']:
        print(f"\n기준 물체 분석:")
        print(f"  기준 물체 사용: 예")
        print(f"  스케일 신뢰도: {ref_analysis['confidence']:.2f}")
    else:
        print(f"\n기준 물체 분석:")
        print(f"  기준 물체 사용: 아니오")
        if ref_analysis.get('suggested_objects'):
            print(f"  추천 기준 물체: {', '.join(ref_analysis['suggested_objects'])}")


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
    
    # 설정 업데이트
    if args.api_key:
        if Config.LLM_PROVIDER == "gemini":
            Config.GEMINI_API_KEY = args.api_key
        else:
            Config.OPENAI_API_KEY = args.api_key
    
    if args.model:
        Config.LLM_MODEL_NAME = args.model
    
    if args.debug:
        Config.DEBUG_MODE = True
    
    if args.simple_debug:
        Config.DEBUG_MODE = True
        Config.SIMPLE_DEBUG = True
    
    if args.no_multimodal:
        Config.ENABLE_MULTIMODAL = False
    
    # 로깅 설정
    setup_logging(Config.DEBUG_MODE, Config.SIMPLE_DEBUG)
    logger = logging.getLogger(__name__)
    
    # 설정 검증
    errors = Config.validate_settings()
    if errors:
        print("설정 오류:")
        for key, error in errors.items():
            print(f"  {key}: {error}")
        return
    
    try:
        # 파이프라인 초기화
        logger.info("파이프라인 초기화 중...")
        pipeline = MassEstimationPipeline(
            yolo_model_path=Config.YOLO_MODEL_PATH,
            midas_model_type=Config.MIDAS_MODEL_TYPE,
            llm_provider=Config.LLM_PROVIDER,
            llm_model_name=Config.LLM_MODEL_NAME,
            multimodal_model_name=Config.MULTIMODAL_MODEL_NAME,
            api_key=Config.GEMINI_API_KEY if Config.LLM_PROVIDER == "gemini" else Config.OPENAI_API_KEY,
            enable_multimodal=Config.ENABLE_MULTIMODAL,
            debug=Config.DEBUG_MODE
        )
        
        # 이미지 처리
        if args.image_path: # Changed from args.batch to args.image_path
            # 단일 이미지 처리
            image_path = args.image_path[0]
            logger.info(f"단일 이미지 처리: {image_path}")
            
            # 입력 검증
            is_valid, error_msg = pipeline.validate_input(image_path)
            if not is_valid:
                print(f"입력 오류: {error_msg}")
                return
            
            # 질량 추정
            result = pipeline.estimate_mass(image_path, Config.SAVE_RESULTS)
            
            # 결과 출력
            print_result_summary(result)
            
            # 시각화
            if args.output: # Changed from args.visualize to args.output
                output_path = args.output or f"visualized_{os.path.basename(image_path)}"
                pipeline.visualize_results(result, output_path)
                print(f"\n시각화 결과 저장: {output_path}")
            
            # 결과 파일 저장
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                print(f"\n결과 파일 저장: {args.output}")
        
        # 파이프라인 통계
        if Config.DEBUG_MODE:
            stats = pipeline.get_pipeline_statistics()
            print(f"\n파이프라인 통계:")
            print(f"  총 처리 횟수: {stats['total_estimations']}")
            print(f"  평균 처리 시간: {stats['average_processing_time']:.2f}초")
            print(f"  평균 신뢰도: {stats['average_confidence']:.2f}")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        print(f"오류: {e}")
        if Config.DEBUG_MODE:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 