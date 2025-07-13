# 🍽️ AI 기반 음식 질량 추정 시스템 (YOLO + MiDaS + LLM)

딥러닝 기반 컴퓨터 비전과 대규모 언어 모델을 결합하여 음식 이미지에서 정확한 질량을 추정하는 시스템입니다.

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 특징](#-주요-특징)
- [시스템 요구사항](#-시스템-요구사항)
- [설치 방법](#-설치-방법)
- [설정 방법](#-설정-방법)
- [사용 방법](#-사용-방법)
- [파이프라인 구조](#-파이프라인-구조)
- [파일 구조](#-파일-구조)
- [결과 해석](#-결과-해석)
- [디버깅 및 트러블슈팅](#-디버깅-및-트러블슈팅)
- [개발자 정보](#-개발자-정보)

## 🎯 프로젝트 개요

이 시스템은 음식 사진에서 자동으로 질량을 추정하는 AI 기반 도구입니다. 다음과 같은 기술들을 통합하여 구현되었습니다:

- **YOLO Segmentation**: 음식과 기준 물체 분할
- **MiDaS Depth Estimation**: 깊이 정보 추정
- **Large Language Model**: 최종 질량 계산 및 검증

### 작동 원리

1. 🎯 **객체 분할**: YOLO 모델로 음식과 기준 물체(이어폰 케이스 등)를 분할
2. 📏 **깊이 추정**: MiDaS 모델로 각 객체의 깊이 정보 계산
3. 🔍 **특징 추출**: 픽셀 면적, 깊이 분포, 상대적 크기 등 계산
4. 📐 **스케일 보정**: 기준 물체를 활용하여 실제 크기 계산
5. 🧠 **LLM 질량 추정**: 추출된 특징을 기반으로 최종 질량 계산

## ✨ 주요 특징

### 🎯 정확한 질량 추정
- **기준 물체 기반 스케일링**: 이어폰 케이스 등 실제 크기를 알 수 있는 물체로 정확한 스케일 계산
- **카메라 정보 활용**: EXIF 데이터에서 초점거리 추출하여 거리 추정 개선
- **적응형 질량 추정**: 단순 부피 계산부터 복잡한 형태 분석까지 다양한 방법 지원

### 🛠️ 유연한 설정
- **다중 LLM 지원**: Gemini, OpenAI GPT 지원
- **다양한 모델 옵션**: YOLO 모델과 MiDaS 버전 선택 가능
- **설정 가능한 파라미터**: 신뢰도 임계값, 이미지 크기 등 조정

### 🔧 개발자 친화적
- **상세한 디버그 모드**: 각 단계별 상세 정보 출력
- **시각화 지원**: 세그멘테이션과 깊이 맵 시각화
- **강건한 오류 처리**: 예외 상황에 대한 fallback 메커니즘

## 💻 시스템 요구사항

### 하드웨어 요구사항
- **RAM**: 최소 8GB, 권장 16GB 이상
- **GPU**: CUDA 지원 GPU (선택사항, 속도 향상)
- **저장공간**: 최소 5GB (모델 파일 포함)

### 소프트웨어 요구사항
- **Python**: 3.8 이상 (권장: 3.9+)
- **운영체제**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **CUDA**: 11.0 이상 (GPU 사용 시)
- **Git LFS**: 대용량 모델 파일 관리용

## 🚀 설치 방법

### 1. 저장소 클론

⚠️ **중요**: 이 프로젝트는 Git LFS를 사용하여 대용량 모델 파일을 관리합니다.

```bash
# Git LFS 설치 (아직 설치되지 않은 경우)
# Windows: https://git-lfs.github.io/ 에서 다운로드
# macOS: brew install git-lfs
# Ubuntu: sudo apt install git-lfs

# 저장소 클론
git clone https://github.com/your-username/Yolo_midas.git
cd Yolo_midas

# Git LFS 파일 다운로드 (모델 파일 포함)
git lfs pull
```

### 2. 가상환경 생성 및 활성화
```bash
# Python venv 사용
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 또는 conda 사용
conda create -n food-estimation python=3.9
conda activate food-estimation
```

### 3. 의존성 설치
```bash
# pip 사용
pip install -r requirements.txt

# 또는 uv 사용 (더 빠름)
uv pip install -r requirements.txt
```

### 4. 모델 파일 확인
```bash
# YOLO 모델 파일 확인
ls -la yolo_food.pt

# 파일이 없는 경우 Git LFS로 다운로드
git lfs pull
```

## ⚙️ 설정 방법

### 1. 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가:

```env
# API 키 설정 (둘 중 하나 선택)
GEMINI_API_KEY=your_gemini_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here

# LLM 설정
LLM_PROVIDER=gemini  # "gemini" 또는 "openai"
LLM_MODEL_NAME=gemini-1.5-flash
MULTIMODAL_MODEL_NAME=gemini-1.5-flash

# 모델 경로 설정
YOLO_MODEL_PATH=yolo_food.pt
MIDAS_MODEL_TYPE=DPT_Large  # "DPT_Large", "DPT_Hybrid", "MiDaS_small"

# 파이프라인 설정
ENABLE_MULTIMODAL=true
DEBUG_MODE=false
SIMPLE_DEBUG=false
SAVE_RESULTS=true

# 이미지 처리 설정
MAX_IMAGE_SIZE=1920
CONFIDENCE_THRESHOLD=0.5

# 디렉토리 설정
RESULTS_DIR=results
LOGS_DIR=logs
DATA_DIR=data
```

### 2. API 키 획득

#### Gemini API 키 (권장)
1. [Google AI Studio](https://makersuite.google.com/app/apikey)에서 API 키 생성
2. 무료 할당량이 충분하며 성능이 우수함

#### OpenAI API 키
1. [OpenAI Platform](https://platform.openai.com/api-keys)에서 API 키 생성
2. 유료 서비스이므로 사용량에 따라 비용 발생

### 3. 설정 확인
```bash
python main.py --config
```

## 🎮 사용 방법

### 기본 사용법
```bash
# 단일 이미지 질량 추정
python main.py path/to/your/image.jpg

# 예시
python main.py data/test1.jpg
```

### 명령줄 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--debug` | 디버그 모드 활성화 | `python main.py image.jpg --debug` |
| `--simple-debug` | 간단 디버그 모드 | `python main.py image.jpg --simple-debug` |
| `--no-multimodal` | 멀티모달 검증 비활성화 | `python main.py image.jpg --no-multimodal` |
| `--output` | 결과 저장 경로 지정 | `python main.py image.jpg --output result.json` |
| `--api-key` | API 키 직접 지정 | `python main.py image.jpg --api-key YOUR_KEY` |
| `--model` | LLM 모델 지정 | `python main.py image.jpg --model gpt-4` |
| `--config` | 현재 설정 확인 | `python main.py --config` |

### 사용 예시

```bash
# 1. 기본 실행
python main.py data/test1.jpg

# 2. 디버그 모드로 실행 (모든 단계 시각화)
python main.py data/test1.jpg --debug

# 3. 간단 디버그 모드 (핵심 정보만)
python main.py data/test1.jpg --simple-debug

# 4. 결과를 JSON 파일로 저장
python main.py data/test1.jpg --output results/analysis.json

# 5. 멀티모달 검증 없이 빠른 처리
python main.py data/test1.jpg --no-multimodal

# 6. 특정 모델 사용
python main.py data/test1.jpg --model gemini-1.5-pro
```

## 🔄 파이프라인 구조

### 전체 흐름도
```
입력 이미지 → YOLO 분할 → MiDaS 깊이 → 특징 추출 → LLM 분석 → 질량 결과
     ↓           ↓           ↓           ↓          ↓
   이미지 로드  음식/기준물체  깊이 맵    픽셀-실제   최종 질량
               세그멘테이션   생성       크기 변환    추정
```

### 상세 단계 설명

#### 1단계: YOLO 세그멘테이션
- **목적**: 음식과 기준 물체 분할
- **출력**: 각 객체의 마스크, 바운딩 박스, 신뢰도
- **시각화**: `results/segmentation_*.jpg`

#### 2단계: MiDaS 깊이 추정
- **목적**: 각 픽셀의 깊이 정보 계산
- **출력**: 정규화된 깊이 맵
- **시각화**: `results/depth_*.jpg`

#### 3단계: 특징 추출
- **목적**: 세그멘테이션과 깊이 정보 결합
- **계산 내용**:
  - 픽셀 면적 → 실제 면적 변환
  - 깊이 분포 → 부피 추정
  - 기준 물체 → 스케일 보정

#### 4단계: 기준 물체 분석
- **목적**: 실제 크기 스케일 계산
- **지원 기준 물체**: 이어폰 케이스, 동전 등
- **출력**: 픽셀-센티미터 변환 비율

#### 5단계: LLM 질량 추정
- **목적**: 최종 질량 계산 및 검증
- **입력**: 추출된 모든 특징
- **출력**: 그램 단위 질량, 신뢰도, 추정 근거

## 📁 파일 구조

```
Yolo_midas/
├── main.py                 # 메인 실행 파일
├── requirements.txt        # Python 의존성
├── pyproject.toml         # 프로젝트 설정
├── uv.lock               # 의존성 락 파일
├── .env                  # 환경 변수 (생성 필요)
├── .gitattributes        # Git LFS 설정
├── .gitignore            # Git 제외 파일
├── README.md             # 이 파일
├── yolo_food.pt          # YOLO 모델 파일 (Git LFS)
│
├── data/                 # 입력 데이터
│   ├── test1.jpg         # 샘플 이미지
│   ├── test5.jpg         # 샘플 이미지
│   └── reference_objects.json  # 기준 물체 정보
│
├── models/              # AI 모델 클래스
│   ├── yolo_model.py    # YOLO 세그멘테이션
│   ├── midas_model.py   # MiDaS 깊이 추정
│   └── llm_model.py     # LLM 질량 추정
│
├── core/                # 핵심 서비스
│   └── estimation_service.py    # 질량 추정 서비스
│
├── api/                 # REST API 서버
│   ├── main.py          # FastAPI 애플리케이션
│   ├── endpoints.py     # API 엔드포인트
│   └── schemas.py       # 데이터 스키마
│
├── config/              # 설정 관리
│   └── settings.py      # 중앙화된 설정
│
├── utils/               # 유틸리티 함수
│   ├── feature_extraction.py    # 특징 추출
│   ├── camera_info_extractor.py # 카메라 정보
│   ├── reference_objects.py     # 기준 물체 관리
│   └── debug_helper.py          # 디버깅 도우미
│
├── results/             # 결과 저장소 (자동 생성)
│   ├── segmentation_*.jpg     # 세그멘테이션 시각화
│   ├── depth_*.jpg           # 깊이 맵 시각화
│   └── mass_estimation_*.json # 질량 추정 결과
│
└── logs/               # 로그 파일 (자동 생성)
    └── main.log        # 실행 로그
```

## 📊 결과 해석

### 출력 형식
```json
{
  "image_path": "data/test1.jpg",
  "processing_time": 5.23,
  "segmentation_results": {
    "food_objects": [
      {
        "class_name": "food",
        "confidence": 0.87,
        "bbox": [100, 150, 300, 400],
        "pixel_area": 25000
      }
    ],
    "reference_objects": [
      {
        "class_name": "earphone_case",
        "confidence": 0.92,
        "bbox": [50, 50, 120, 180],
        "pixel_area": 9100
      }
    ]
  },
  "initial_estimate": {
    "estimated_mass": 120.5,
    "confidence": 0.75,
    "method": "volume_based"
  },
  "final_estimate": {
    "final_mass": 118.3,
    "confidence": 0.82,
    "method": "multimodal_verified",
    "reasoning": "이어폰 케이스를 기준으로 한 스케일 보정 결과..."
  }
}
```

### 결과 해석 가이드

#### 신뢰도 점수
- **0.8 이상**: 매우 높음 (신뢰 가능)
- **0.6-0.8**: 높음 (일반적으로 신뢰 가능)
- **0.4-0.6**: 보통 (참고용)
- **0.4 미만**: 낮음 (부정확할 가능성)

#### 추정 방법
- **`volume_based`**: 부피 기반 계산
- **`reference_scaled`**: 기준 물체 스케일 적용
- **`multimodal_verified`**: LLM 멀티모달 검증
- **`fallback`**: 기본값 사용

### 시각화 파일

#### 세그멘테이션 시각화
- **파일**: `results/segmentation_*.jpg`
- **내용**: 분할된 객체들의 마스크와 바운딩 박스
- **색상**: 음식(빨간색), 기준 물체(파란색)

#### 깊이 맵 시각화
- **파일**: `results/depth_*.jpg`
- **내용**: 깊이 정보와 객체 경계
- **색상**: 가까운 곳(밝음), 먼 곳(어두움)

## 🔧 디버깅 및 트러블슈팅

### 일반적인 문제들

#### 1. API 키 오류
```bash
# 오류 메시지
ERROR: Gemini API 키가 설정되지 않았습니다

# 해결 방법
1. .env 파일에 API 키 추가
2. 또는 명령줄에서 직접 지정: --api-key YOUR_KEY
```

#### 2. 모델 파일 없음
```bash
# 오류 메시지
ERROR: YOLO 모델 파일이 존재하지 않습니다

# 해결 방법
1. Git LFS로 모델 파일 다운로드: git lfs pull
2. 파일 존재 확인: ls -la yolo_food.pt
```

#### 3. Git LFS 문제
```bash
# Git LFS 파일이 다운로드되지 않은 경우
git lfs pull

# Git LFS 상태 확인
git lfs ls-files

# Git LFS 재설치
git lfs install
```

#### 4. 메모리 부족
```bash
# 해결 방법
1. MAX_IMAGE_SIZE를 줄임 (예: 1920 → 1280)
2. 더 가벼운 모델 사용: MiDaS_small
3. 배치 크기 줄임: BATCH_SIZE=1
```

#### 5. GPU 사용 불가
```bash
# 확인 방법
python -c "import torch; print(torch.cuda.is_available())"

# 해결 방법
1. CUDA 드라이버 업데이트
2. PyTorch GPU 버전 설치
3. 또는 CPU 모드로 실행 (느리지만 작동)
```

### 디버그 모드 활용

#### 상세 디버그 모드
```bash
python main.py image.jpg --debug
```
- 모든 단계 정보 출력
- 중간 결과 시각화
- 성능 통계 표시

#### 간단 디버그 모드
```bash
python main.py image.jpg --simple-debug
```
- 핵심 정보만 출력
- 외부 라이브러리 로그 숨김
- 빠른 실행

### 로그 확인
```bash
# 실행 로그
tail -f logs/main.log

# 질량 추정 로그
tail -f mass_estimation.log
```

## 🤝 개발자 정보

### 기여 방법
1. 이슈 리포트: 버그나 개선사항 제안
2. 풀 리퀘스트: 코드 개선 기여
3. 문서 개선: README나 주석 개선

### 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다.

### 연락처
- GitHub: [프로젝트 저장소](https://github.com/your-username/Yolo_midas)
- 이슈 트래커: [GitHub Issues](https://github.com/your-username/Yolo_midas/issues)

---

**🎉 프로젝트가 도움이 되셨다면 ⭐ 스타를 눌러주세요!** 