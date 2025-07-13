"""
공통 로깅 설정 유틸리티
"""

import logging
import warnings
from typing import Optional


def setup_logging(
    debug: bool = False, 
    simple_debug: bool = False,
    log_level: Optional[str] = None
) -> None:
    """
    애플리케이션 전체의 로깅을 설정합니다.
    
    Args:
        debug: 디버그 모드 (모든 로그 출력)
        simple_debug: 간단 디버그 모드 (WARNING 이상만 출력)
        log_level: 로그 레벨 (INFO, DEBUG, WARNING, ERROR)
    """
    if simple_debug:
        level = logging.WARNING
        _suppress_external_logs()
        _suppress_warnings()
    elif debug:
        level = logging.DEBUG
    else:
        level = getattr(logging, log_level.upper()) if log_level else logging.INFO
    
    # 기본 로깅 설정
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 주요 모듈 로그 레벨 설정
    if simple_debug:
        _set_main_module_levels()


def _suppress_external_logs() -> None:
    """외부 라이브러리 로그 비활성화"""
    external_loggers = [
        'PIL', 'PIL.TiffImagePlugin',
        'ultralytics', 'ultralytics.yolo.utils', 'ultralytics.yolo.v8',
        'torch', 'torch.hub',
        'timm', 'timm.models',
        'transformers',
        'urllib3', 'requests'
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def _suppress_warnings() -> None:
    """Python warnings 비활성화"""
    warnings.filterwarnings('ignore')


def _set_main_module_levels() -> None:
    """주요 모듈 로그 레벨 설정"""
    main_modules = [
        'root', 'core.estimation_service', '__main__',
        'models.yolo_model', 'models.midas_model', 'models.llm_model'
    ]
    
    for module_name in main_modules:
        logging.getLogger(module_name).setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    표준화된 로거를 반환합니다.
    
    Args:
        name: 로거 이름
        
    Returns:
        설정된 로거 객체
    """
    return logging.getLogger(name) 