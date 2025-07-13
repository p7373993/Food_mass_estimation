"""
모든 AI 모델의 공통 베이스 클래스
"""

import logging
import torch
from abc import ABC, abstractmethod, ABCMeta
from typing import TypeVar, Generic, Optional, Any
from config.settings import settings

T = TypeVar('T')

class SingletonABCMeta(ABCMeta):
    """싱글톤 + ABC 메타클래스"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class BaseModel(ABC, metaclass=SingletonABCMeta):
    """
    모든 AI 모델의 공통 베이스 클래스
    - 싱글톤 패턴 자동 적용
    - 공통 초기화 로직
    - 에러 처리 표준화
    """
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._model: Optional[Any] = None
            self._device = self._get_device()
            self._initialize_model()
            self._initialized = True
    
    def _get_device(self) -> str:
        """디바이스 선택 로직"""
        return 'cuda' if torch.cuda.is_available() and settings.USE_GPU else 'cpu'
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """모델 초기화 (각 모델에서 구현)"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """모델 이름 반환 (로깅용)"""
        pass
    
    def _log_success(self, message: str) -> None:
        """성공 로그"""
        logging.info(f"{self.get_model_name()} {message}")
    
    def _log_error(self, message: str, error: Exception) -> None:
        """에러 로그"""
        logging.error(f"{self.get_model_name()} {message}: {error}")
    
    def is_ready(self) -> bool:
        """모델이 사용 가능한지 확인"""
        return self._model is not None
    
    @property
    def device(self) -> str:
        """디바이스 정보"""
        return self._device
    
    @property
    def model(self) -> Any:
        """모델 객체"""
        return self._model 