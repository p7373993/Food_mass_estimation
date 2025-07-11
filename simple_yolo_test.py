import cv2
import numpy as np
from ultralytics import YOLO
import os

print(f"현재 작업 디렉토리: {os.getcwd()}")

# --- ⚙️ 설정 (사용자 환경에 맞게 수정) ---

# 1. 로컬에 저장된 최종 모델 경로
MODEL_PATH = 'weights/yolo_food_v1.pt'

# 2. 확인할 이미지 파일의 경로
IMAGE_PATH = 'data/20250701_111358.jpg'

# 3. 결과 이미지를 저장할 경로
OUTPUT_PATH = 'results/simple_yolo_test_output.jpg'

# --- 코드 시작 ---

print("--- 단순 YOLO 세그멘테이션 테스트 시작 ---")

# 모델 로드
print(f"[1/4] 모델을 로드합니다: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    exit()
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    raise

# 이미지 예측 실행
# 신뢰도를 0.1로 낮춰서 테스트
print(f"[2/4] 이미지를 분석합니다 (신뢰도 0.1): {IMAGE_PATH}")
if not os.path.exists(IMAGE_PATH):
    print(f"❌ 이미지 파일을 찾을 수 없습니다: {IMAGE_PATH}")
    exit()
results = model(IMAGE_PATH, conf=0.1)

# 원본 이미지 로드
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"❌ OpenCV가 이미지를 로드하지 못했습니다: {IMAGE_PATH}")
    exit()

print("[3/4] 결과 분석 및 시각화 중...")
# 결과 분석 및 시각화
food_found = False
# results[0].masks가 비어있지 않은 경우에만 루프 실행
if results[0].masks is not None:
    for seg, box in zip(results[0].masks.xy, results[0].boxes):
        # 클래스 이름 확인
        class_name = model.names[int(box.cls)]
        confidence = float(box.conf)

        # ✨ 'food' 클래스일 경우에만 마스크를 그림 ✨
        if class_name == 'food':
            food_found = True
            print(f"  -> ✅ 'food' 객체 발견! (신뢰도: {confidence:.2f})")

            # 마스크 그리기 (반투명 파란색)
            overlay = image.copy()
            cv2.fillPoly(overlay, [seg.astype(np.int32)], color=(255, 0, 0)) # BGR 순서로 파란색
            alpha = 0.4 # 투명도
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # 경계선 그리기 (더 잘 보이게)
            cv2.polylines(image, [seg.astype(np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)

if not food_found:
    print("  -> ⚠️ 이 이미지에서는 'food' 객체를 찾지 못했습니다.")

# 최종 결과 이미지 파일로 저장
# 저장 전 디렉토리 확인
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
cv2.imwrite(OUTPUT_PATH, image)
print(f"[4/4] 결과 이미지를 저장했습니다: {OUTPUT_PATH}")
print("\n--- 테스트 완료 ---") 