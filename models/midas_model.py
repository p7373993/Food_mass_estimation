import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import urllib.request
import os
from PIL import Image, ImageOps
from pipeline.config import Config

class MiDaSDepthModel:
    """
    MiDaS 모델을 사용하여 단일 이미지에서 깊이 정보를 추정하는 클래스
    """
    
    def __init__(self, model_type: str = "DPT_Large"):
        """
        MiDaS 모델 초기화
        
        Args:
            model_type: 사용할 MiDaS 모델 타입 ("DPT_Large", "DPT_Hybrid", "MiDaS_small")
        """
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델별 입력 크기 설정
        default_size = Config.MIDAS_INPUT_SIZE
        self.input_sizes = {
            "DPT_Large": (default_size, default_size),
            "DPT_Hybrid": (default_size, default_size),
            "MiDaS_small": (256, 256)  # 작은 모델만 256 유지
        }
        
        self.input_size = self.input_sizes.get(model_type, (default_size, default_size))
        
        self.load_model()
    
    def load_model(self):
        """MiDaS 모델 로드"""
        try:
            # 간단 디버그 모드에서 torch.hub 출력 숨기기
            if hasattr(Config, 'SIMPLE_DEBUG') and Config.SIMPLE_DEBUG:
                import sys
                import os
                from contextlib import redirect_stdout
                
                # stdout을 잠시 /dev/null로 redirect
                with open(os.devnull, 'w') as devnull:
                    with redirect_stdout(devnull):
                        # PyTorch Hub에서 MiDaS 모델 로드
                        self.model = torch.hub.load('intel-isl/MiDaS', self.model_type, verbose=False)
            else:
                # 일반 모드에서는 정상 출력
                self.model = torch.hub.load('intel-isl/MiDaS', self.model_type)
            
            self.model.to(self.device)
            self.model.eval()
            
            # 전처리 변환 설정
            self.transform = Compose([
                Resize(self.input_size),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logging.info(f"MiDaS 모델이 성공적으로 로드되었습니다: {self.model_type}")
            
        except Exception as e:
            logging.error(f"MiDaS 모델 로드 실패: {e}")
            raise
    
    def estimate_depth(self, image_path: str) -> Dict:
        """
        이미지에서 깊이 추정 수행
        
        Args:
            image_path: 입력 이미지 경로
            
        Returns:
            깊이 추정 결과 딕셔너리
        """
        try:
            # 이미지 읽기 (PIL로 읽기) - EXIF 회전 정보 처리
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image)  # EXIF 회전 정보 처리
            image = image.convert('RGB')
            original_shape = np.array(image).shape
            
            # 전처리
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 깊이 추정
            with torch.no_grad():
                depth = self.model(input_tensor)
                
                # 원본 이미지 크기로 리사이즈
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=(original_shape[0], original_shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            
            # 깊이 맵을 numpy 배열로 변환
            depth_map = depth.cpu().numpy()
            
            # 깊이 맵 정규화 (0-255 범위로)
            depth_normalized = self._normalize_depth(depth_map)
            
            return {
                "depth_map": depth_map,
                "depth_normalized": depth_normalized,
                "original_shape": original_shape,
                "depth_stats": self._calculate_depth_stats(depth_map)
            }
            
        except Exception as e:
            logging.error(f"깊이 추정 중 오류: {e}")
            raise
    
    def _normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        깊이 맵을 0-255 범위로 정규화
        
        Args:
            depth_map: 원본 깊이 맵
            
        Returns:
            정규화된 깊이 맵
        """
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min) * 255
        else:
            depth_normalized = np.zeros_like(depth_map)
        
        return depth_normalized.astype(np.uint8)
    
    def _calculate_depth_stats(self, depth_map: np.ndarray) -> Dict:
        """
        깊이 맵의 통계 정보 계산
        
        Args:
            depth_map: 깊이 맵
            
        Returns:
            깊이 통계 딕셔너리
        """
        return {
            "min_depth": float(depth_map.min()),
            "max_depth": float(depth_map.max()),
            "mean_depth": float(depth_map.mean()),
            "std_depth": float(depth_map.std()),
            "median_depth": float(np.median(depth_map))
        }
    
    def get_object_depth_info(self, depth_map: np.ndarray, mask: np.ndarray) -> Dict:
        """
        특정 객체(마스크)의 깊이 정보 추출
        
        Args:
            depth_map: 깊이 맵
            mask: 객체 마스크
            
        Returns:
            객체의 깊이 정보 딕셔너리
        """
        try:
            # 마스크 영역의 깊이 값만 추출
            object_depths = depth_map[mask > 0]
            
            if len(object_depths) == 0:
                return {
                    "min_depth": 0.0,
                    "max_depth": 0.0,
                    "mean_depth": 0.0,
                    "std_depth": 0.0,
                    "median_depth": 0.0,
                    "depth_variation": 0.0
                }
            
            # 깊이 통계 계산
            min_depth = float(object_depths.min())
            max_depth = float(object_depths.max())
            mean_depth = float(object_depths.mean())
            std_depth = float(object_depths.std())
            median_depth = float(np.median(object_depths))
            depth_variation = max_depth - min_depth
            
            return {
                "min_depth": min_depth,
                "max_depth": max_depth,
                "mean_depth": mean_depth,
                "std_depth": std_depth,
                "median_depth": median_depth,
                "depth_variation": depth_variation
            }
            
        except Exception as e:
            logging.error(f"객체 깊이 정보 추출 중 오류: {e}")
            raise
    
    def estimate_relative_size(self, depth_map: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> Dict:
        """
        두 객체 간의 상대적 크기 추정
        
        Args:
            depth_map: 깊이 맵
            mask1: 첫 번째 객체 마스크
            mask2: 두 번째 객체 마스크
            
        Returns:
            상대적 크기 정보 딕셔너리
        """
        try:
            # 각 객체의 깊이 정보 추출
            obj1_depth = self.get_object_depth_info(depth_map, mask1)
            obj2_depth = self.get_object_depth_info(depth_map, mask2)
            
            # 픽셀 면적 계산
            area1 = np.sum(mask1)
            area2 = np.sum(mask2)
            
            # 평균 깊이 비율 계산 (깊이가 작을수록 가까움)
            depth_ratio = obj1_depth["mean_depth"] / obj2_depth["mean_depth"] if obj2_depth["mean_depth"] != 0 else 1.0
            
            # 거리 보정된 면적 비율 계산
            # 가까운 객체일수록 실제보다 크게 보임
            corrected_area_ratio = (area1 / area2) * (depth_ratio ** 2) if area2 != 0 else 1.0
            
            return {
                "pixel_area_ratio": area1 / area2 if area2 != 0 else 1.0,
                "depth_ratio": depth_ratio,
                "corrected_area_ratio": corrected_area_ratio,
                "object1_depth": obj1_depth,
                "object2_depth": obj2_depth
            }
            
        except Exception as e:
            logging.error(f"상대적 크기 추정 중 오류: {e}")
            raise
    
    def visualize_depth(self, image_path: str, save_path: str = None, segmentation_results: Dict = None, 
                       show_colorbar: bool = True, show_stats: bool = True) -> np.ndarray:
        """
        깊이 맵을 시각화 (개선된 버전)
        
        Args:
            image_path: 입력 이미지 경로
            save_path: 저장할 경로 (선택사항)
            segmentation_results: 세그멘테이션 결과 (선택사항)
            show_colorbar: 컬러바 표시 여부
            show_stats: 통계 정보 표시 여부
            
        Returns:
            시각화된 깊이 맵
        """
        try:
            # 깊이 추정
            depth_results = self.estimate_depth(image_path)
            depth_map = depth_results["depth_map"]
            depth_normalized = depth_results["depth_normalized"]
            depth_stats = depth_results["depth_stats"]
            
            # 컬러맵 적용 (JET 컬러맵 사용)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # 원본 이미지 로드 - EXIF 정보 처리
            pil_image = Image.open(image_path)
            pil_image = ImageOps.exif_transpose(pil_image)  # EXIF 회전 정보 처리
            original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 이미지 크기 확인 및 조정
            h, w = original_image.shape[:2]
            
            # 깊이 맵 크기를 원본 이미지 크기에 맞춤
            if depth_colormap.shape[:2] != (h, w):
                depth_colormap = cv2.resize(depth_colormap, (w, h))
            
            # 시각화 이미지 생성 (깊이맵만 표시)
            vis_image = depth_colormap.copy()
            
            # 세그멘테이션 결과가 있으면 객체별 깊이 정보 표시
            if segmentation_results:
                # 음식 객체들 분석
                for i, food_obj in enumerate(segmentation_results.get("food_objects", [])):
                    mask = food_obj.get("mask", np.zeros((h, w)))
                    # 마스크를 깊이 맵 크기에 맞춤
                    if mask.shape != depth_map.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), 
                                        (depth_map.shape[1], depth_map.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                    
                    # 객체별 깊이 정보 추출
                    obj_depth_info = self.get_object_depth_info(depth_map, mask)
                    
                    # 객체 중심점 계산
                    bbox = food_obj.get("bbox", [0, 0, w, h])
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    
                    # 깊이 정보 텍스트 표시
                    depth_text = f"Food #{i+1}: {obj_depth_info['mean_depth']:.1f}"
                    cv2.putText(vis_image, depth_text, (center_x - 50, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # 객체 윤곽선 표시 (시각화용 마스크는 원본 크기로)
                    vis_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(vis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_image, contours, -1, (255, 255, 255), 2)
                
                # 기준 객체들 분석
                for i, ref_obj in enumerate(segmentation_results.get("reference_objects", [])):
                    mask = ref_obj.get("mask", np.zeros((h, w)))
                    # 마스크를 깊이 맵 크기에 맞춤
                    if mask.shape != depth_map.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), 
                                        (depth_map.shape[1], depth_map.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                    
                    # 객체별 깊이 정보 추출
                    obj_depth_info = self.get_object_depth_info(depth_map, mask)
                    
                    # 객체 중심점 계산
                    bbox = ref_obj.get("bbox", [0, 0, w, h])
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    
                    # 깊이 정보 텍스트 표시
                    depth_text = f"Ref #{i+1}: {obj_depth_info['mean_depth']:.1f}"
                    cv2.putText(vis_image, depth_text, (center_x - 50, center_y + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # 객체 윤곽선 표시 (시각화용 마스크는 원본 크기로)
                    vis_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(vis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_image, contours, -1, (0, 255, 255), 2)
            
            # 제목 추가
            cv2.putText(vis_image, "Depth Map", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 통계 정보 표시
            if show_stats:
                stats_y = 70
                cv2.putText(vis_image, f"Depth Range: {depth_stats['min_depth']:.1f} - {depth_stats['max_depth']:.1f}", 
                           (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis_image, f"Mean: {depth_stats['mean_depth']:.1f}", 
                           (10, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis_image, f"Std: {depth_stats['std_depth']:.1f}", 
                           (10, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 컬러바 표시 (간단한 버전)
            if show_colorbar:
                # 세로 컬러바 생성
                colorbar_height = 200
                colorbar_width = 20
                colorbar_x = w - 80  # 라벨을 위한 충분한 여백
                colorbar_y = h - colorbar_height - 50
                
                # 컬러바 생성
                colorbar_gradient = np.linspace(0, 255, colorbar_height, dtype=np.uint8)
                colorbar_gradient = np.repeat(colorbar_gradient.reshape(-1, 1), colorbar_width, axis=1)
                colorbar_colored = cv2.applyColorMap(colorbar_gradient, cv2.COLORMAP_JET)
                
                # 컬러바를 이미지에 삽입
                vis_image[colorbar_y:colorbar_y + colorbar_height, colorbar_x:colorbar_x + colorbar_width] = colorbar_colored
                
                # 컬러바 라벨 (컬러바 왼쪽에 배치)
                cv2.putText(vis_image, "Near", (colorbar_x - 50, colorbar_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(vis_image, "Far", (colorbar_x - 40, colorbar_y + colorbar_height - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 이미지 저장
            if save_path:
                # 디렉토리가 없으면 생성
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                cv2.imwrite(save_path, vis_image)
                logging.info(f"깊이 맵 시각화 저장: {save_path}")
            
            return vis_image
            
        except Exception as e:
            logging.error(f"깊이 시각화 중 오류: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_type": self.model_type,
            "device": str(self.device),
            "input_size": self.input_size
        } 