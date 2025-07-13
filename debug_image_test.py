import cv2
import numpy as np
import base64
from pathlib import Path

def test_image_processing():
    """멀티모달 검증에서 이미지가 어떻게 처리되는지 테스트"""
    
    # 원본 이미지 로드
    image_path = "data/test1.jpg"
    image = cv2.imread(image_path)
    
    print(f"원본 이미지 정보:")
    print(f"  크기: {image.shape}")
    print(f"  BGR 평균: {np.mean(image, axis=(0,1))}")
    print(f"  RGB 평균: {np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=(0,1))}")
    
    # 멀티모달 검증에서 사용하는 이미지 처리 과정
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    max_size = 1024
    h, w = image_rgb.shape[:2]
    
    print(f"\n리사이즈 전:")
    print(f"  크기: {image_rgb.shape}")
    print(f"  RGB 평균: {np.mean(image_rgb, axis=(0,1))}")
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        print(f"\n리사이즈 후:")
        print(f"  스케일: {scale:.3f}")
        print(f"  새 크기: {image_rgb.shape}")
        print(f"  RGB 평균: {np.mean(image_rgb, axis=(0,1))}")
    
    # JPEG 인코딩 (BGR로 변환해서 인코딩)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    print(f"\n인코딩 정보:")
    print(f"  JPEG 버퍼 크기: {len(buffer)} bytes")
    print(f"  Base64 길이: {len(image_base64)} 문자")
    
    # 디코딩하여 저장 (실제로 LLM에 전달되는 이미지 확인)
    decoded_buffer = base64.b64decode(image_base64)
    decoded_image = cv2.imdecode(np.frombuffer(decoded_buffer, np.uint8), cv2.IMREAD_COLOR)
    
    print(f"\n디코딩 후:")
    print(f"  크기: {decoded_image.shape}")
    print(f"  BGR 평균: {np.mean(decoded_image, axis=(0,1))}")
    print(f"  RGB 평균: {np.mean(cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB), axis=(0,1))}")
    
    # 저장
    cv2.imwrite("debug_original.jpg", image)
    cv2.imwrite("debug_processed.jpg", decoded_image)
    
    print(f"\n이미지 저장 완료:")
    print(f"  원본: debug_original.jpg")
    print(f"  처리된 이미지: debug_processed.jpg")

if __name__ == "__main__":
    test_image_processing() 