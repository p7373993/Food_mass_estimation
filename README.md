# 🍽️ AI 기반 음식 질량 추정 시스템 (YOLO + MiDaS + LLM)

딥러닝 기반 컴퓨터 비전과 대규모 언어 모델을 결합하여 음식 이미지에서 정확한 질량을 추정하는 시스템입니다.

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 특징](#-주요-특징)
- [시스템 요구사항](#-시스템-요구사항)
- [설치 방법](#-설치-방법)
- [설정 방법](#-설정-방법)
- [사용 방법](#-사용-방법)
- [API 서버 및 WebSocket](#-api-서버-및-websocket)
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
- **FastAPI + WebSocket**: 실시간 API 서버 및 알림 시스템

### 작동 원리

1. 🎯 **객체 분할**: YOLO 모델로 음식과 기준 물체(이어폰 케이스 등)를 분할
2. 📏 **깊이 추정**: MiDaS 모델로 각 객체의 깊이 정보 계산
3. 🔍 **특징 추출**: 픽셀 면적, 깊이 분포, 상대적 크기 등 계산
4. 📐 **스케일 보정**: 기준 물체를 활용하여 실제 크기 계산
5. 🧠 **LLM 질량 추정**: 추출된 특징을 기반으로 최종 질량 계산
6. 🌐 **API 서버**: RESTful API와 WebSocket을 통한 실시간 처리

## ✨ 주요 특징

### 🎯 정확한 질량 추정
- **기준 물체 기반 스케일링**: 이어폰 케이스 등 실제 크기를 알 수 있는 물체로 정확한 스케일 계산
- **부피 기반 계산**: 깊이 정보를 활용한 3D 부피 추정
- **LLM 검증**: 대규모 언어 모델을 통한 최종 질량 검증

### 🛠️ 유연한 설정
- **다중 LLM 지원**: Gemini, OpenAI GPT 지원
- **다양한 모델 옵션**: YOLO 모델과 MiDaS 버전 선택 가능
- **설정 가능한 파라미터**: 신뢰도 임계값, 이미지 크기 등 조정

### 🔧 개발자 친화적
- **상세한 디버그 모드**: 각 단계별 상세 정보 출력
- **시각화 지원**: 세그멘테이션과 깊이 맵 시각화
- **강건한 오류 처리**: 예외 상황에 대한 fallback 메커니즘

### 🌐 API 서버 및 실시간 알림
- **RESTful API**: HTTP 기반 질량 추정 API
- **WebSocket 실시간 알림**: 작업 진행 상황 실시간 모니터링
- **동기/비동기 처리**: 즉시 결과 또는 백그라운드 처리 선택
- **CORS 지원**: 웹 브라우저에서 직접 사용 가능

## 💻 시스템 요구사항

### 하드웨어 요구사항
- **RAM**: 최소 8GB, 권장 16GB 이상
- **GPU**: CUDA 지원 GPU (선택사항, 속도 향상)
- **저장공간**: 최소 5GB (모델 파일 포함)

### 소프트웨어 요구사항
- **Python**: 3.8 이상 (권장: 3.12)
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
ls -la weights/yolo_food_v1.pt

# 파일이 없는 경우 Git LFS로 다운로드
git lfs pull
```

## ⚙️ 설정 방법

### 1. 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가:

```env
# =============================================================================
# API 키 설정 (필수)
# =============================================================================
# Google Gemini API 키
GEMINI_API_KEY=

# OpenAI API 키 (선택사항)
OPENAI_API_KEY=

# =============================================================================
# LLM 모델 설정 (필요시 변경)
# =============================================================================
# LLM 제공자 선택: "gemini" 또는 "openai"
LLM_PROVIDER=gemini

# 사용할 LLM 모델 이름
LLM_MODEL_NAME=gemini-2.5-flash

# 멀티모달 검증용 모델 이름
MULTIMODAL_MODEL_NAME=gemini-2.5-flash

# =============================================================================
# 핵심 동작 설정 (필요시 변경)
# =============================================================================
# 멀티모달 검증 활성화 여부
ENABLE_MULTIMODAL=true

# 디버그 모드 (개발용)
DEBUG_MODE=false

# 결과 저장 여부
SAVE_RESULTS=true

# =============================================================================
# 참고사항
# =============================================================================
# - 다른 모든 설정들은 config/settings.py에서 관리됩니다
# - YOLO 모델 경로, MiDaS 설정, 기본값들은 코드에서 자동 관리
# - 필요한 경우에만 위 설정들을 수정하세요

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

### 1. 명령줄 실행

#### 기본 사용법
```bash
# 단일 이미지 질량 추정
python main.py path/to/your/image.jpg

# 예시
python main.py data/test1.jpg
```
python main.py data/test1.jpg --debug //이거 많이씀씀

#### 명령줄 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--debug` | 디버그 모드 활성화 | `python main.py image.jpg --debug` |
| `--simple-debug` | 간단 디버그 모드 | `python main.py image.jpg --simple-debug` |
| `--no-multimodal` | 멀티모달 검증 비활성화 | `python main.py image.jpg --no-multimodal` |
| `--output` | 결과 저장 경로 지정 | `python main.py image.jpg --output result.json` |
| `--api-key` | Gemini/OpenAI API 키 직접 지정 | `python main.py image.jpg --api-key YOUR_KEY` |
| `--model` | LLM 모델 지정 | `python main.py image.jpg --model gpt-4` | 
| `--config` | 현재 설정 확인 | `python main.py --config` |

#### 사용 예시

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

### 2. API 서버 실행

#### 서버 시작
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
```

#### API 문서
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## 🌐 API 서버 및 WebSocket

### 서버 실행
```bash
# 개발 모드 (자동 재시작)
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8001

# 프로덕션 모드
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### 사용 가능한 엔드포인트

#### 1. 동기 처리 (즉시 결과)
```http
POST /api/v1/estimate
Content-Type: multipart/form-data

# 요청: 이미지 파일 업로드
# 응답: 즉시 질량 추정 결과
```

#### 2. 비동기 처리 (백그라운드 작업)
```http
POST /api/v1/estimate_async
Content-Type: multipart/form-data

# 요청: 이미지 파일 업로드
# 응답: 작업 ID 반환
```

#### 3. 작업 상태 조회
```http
GET /api/v1/task/{task_id}

# 응답: 작업 진행 상황 및 결과
```

#### 4. WebSocket 실시간 알림
```http
WS /api/v1/ws/task/{task_id}

# 실시간으로 작업 진행 상황 수신
```

#### 5. 파이프라인 상태 확인
```http
GET /api/v1/pipeline-status

# 각 모델의 로드 상태 확인
```

#### 6. 서버 상태 확인
```http
GET /health

# 서버 정상 동작 확인
```

### API 사용 예시

#### Python 클라이언트
```python
import requests
import json

# 1. 동기 처리
def estimate_mass_sync(image_path):
    url = "http://localhost:8001/api/v1/estimate"
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    return response.json()

# 2. 비동기 처리
def estimate_mass_async(image_path):
    # 작업 시작
    url = "http://localhost:8001/api/v1/estimate_async"
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    task_id = response.json()['task_id']
    
    # 작업 상태 확인
    status_url = f"http://localhost:8001/api/v1/task/{task_id}"
    while True:
        status_response = requests.get(status_url)
        status_data = status_response.json()
        
        if status_data['status'] == 'completed':
            return status_data['result']
        elif status_data['status'] == 'failed':
            raise Exception(status_data['error'])
        
        time.sleep(1)  # 1초 대기

# 사용 예시
result = estimate_mass_sync('data/test1.jpg')
print(f"추정 질량: {result['mass_estimation']['estimated_mass_g']}g")
```

#### JavaScript 클라이언트
```javascript
// 1. 동기 처리
async function estimateMassSync(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8001/api/v1/estimate', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// 2. 비동기 처리 + WebSocket
async function estimateMassAsync(file) {
    // 작업 시작
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8001/api/v1/estimate_async', {
        method: 'POST',
        body: formData
    });
    
    const { task_id } = await response.json();
    
    // WebSocket 연결
    const ws = new WebSocket(`ws://localhost:8001/api/v1/ws/task/${task_id}`);
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        if (data.type === 'task_status') {
            console.log(`진행률: ${data.data.progress * 100}%`);
        } else if (data.type === 'task_completed') {
            console.log('완료!', data.data.result);
            ws.close();
        }
    };
    
    return task_id;
}

// 사용 예시
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    const result = await estimateMassSync(file);
    console.log(`추정 질량: ${result.mass_estimation.estimated_mass_g}g`);
});
```

### API 응답 형식

#### 동기 처리 응답
```json
{
  "filename": "test1.jpg",
  "detected_objects": {
    "food": 1,
    "reference_objects": 1
  },
  "mass_estimation": {
    "estimated_mass_g": 150.5,
    "confidence": 0.75,
    "food_name": "김밥",
    "verification_method": "volume_based",
    "calculation_details": {
      "volume_cm3": 120.3,
      "density_g_cm3": 1.25,
      "reference_object_used": "이어폰 케이스",
      "pixel_to_cm_ratio": 0.05
    }
  }
}
```

#### 비동기 작업 생성 응답
```json
{
  "task_id": "1af8db93-c3cc-42dd-bd44-54dd68d9abc2",
  "status": "processing",
  "message": "작업이 시작되었습니다.",
  "created_at": "2025-07-14T15:22:40.500031"
}
```

#### 작업 상태 조회 응답
```json
{
  "task_id": "1af8db93-c3cc-42dd-bd44-54dd68d9abc2",
  "status": "completed",
  "progress": 1.0,
  "message": "작업이 완료되었습니다.",
  "created_at": "2025-07-14T15:22:40.500031",
  "completed_at": "2025-07-14T15:23:26.326864",
  "result": {
    "filename": "test1.jpg",
    "detected_objects": {
      "food": 1,
      "reference_objects": 1
    },
    "mass_estimation": {
      "estimated_mass_g": 150.5,
      "confidence": 0.75,
      "food_name": "김밥",
      "verification_method": "volume_based"
    }
  }
}
```

#### WebSocket 메시지 형식
```json
// 진행 상황 업데이트
{
  "type": "task_status",
  "task_id": "1af8db93-c3cc-42dd-bd44-54dd68d9abc2",
  "data": {
    "status": "processing",
    "progress": 0.6,
    "message": "LLM 분석 중...",
    "current_step": "llm_analysis"
  }
}

// 작업 완료
{
  "type": "task_completed",
  "task_id": "1af8db93-c3cc-42dd-bd44-54dd68d9abc2",
  "data": {
    "status": "completed",
    "progress": 1.0,
    "message": "작업이 완료되었습니다.",
    "result": {
      "filename": "test1.jpg",
      "mass_estimation": {
        "estimated_mass_g": 150.5,
        "confidence": 0.75
      }
    }
  }
}
```

### CORS 설정
API 서버는 다음 origin들을 허용합니다:
- `http://localhost:5500` (Live Server)
- `http://127.0.0.1:5500`
- `http://localhost:3000`
- `http://127.0.0.1:3000`
- `*` (개발 환경)

### 에러 처리
```json
{
  "error": "파일 업로드 실패",
  "detail": "지원하지 않는 파일 형식입니다. JPG, PNG 파일만 업로드 가능합니다.",
  "status_code": 400
}
```

### 성능 최적화 팁
1. **이미지 크기**: 1920px 이하 권장
2. **동시 요청**: 서버 리소스에 따라 제한
3. **비동기 처리**: 대용량 이미지나 긴 처리 시간이 필요한 경우
4. **WebSocket**: 실시간 진행 상황이 필요한 경우

### WebSocket 테스트 방법

#### 1. 테스트 HTML 파일 사용
프로젝트에 포함된 `websocket_test.html` 파일을 사용하여 WebSocket 기능을 테스트할 수 있습니다.

```bash
# 1. API 서버 시작
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8001

# 2. Live Server로 HTML 파일 실행
# VS Code에서 websocket_test.html 파일을 열고 "Go Live" 버튼 클릭
# 또는 다른 로컬 서버 사용 (http://localhost:5500)
```

#### 2. 테스트 순서
1. **브라우저에서 `http://localhost:5500/websocket_test.html` 접속**
2. **"WebSocket 연결" 버튼 클릭** - 연결 상태 확인
3. **"파일 선택" 버튼으로 이미지 업로드** - `data/test1.jpg` 또는 `data/test2.jpg` 사용
4. **실시간 진행 상황 확인** - WebSocket을 통한 실시간 알림
5. **최종 결과 확인** - 질량 추정 결과 표시

#### 3. 예상 결과
```
[오후 4:02:07] WebSocket 연결됨
[오후 4:02:07] 파일 업로드 시작: test1.jpg (2310742 bytes)
[오후 4:02:08] 작업 시작됨
[오후 4:02:09] YOLO 분석 중... (진행률: 20%)
[오후 4:02:10] MiDaS 깊이 추정 중... (진행률: 40%)
[오후 4:02:11] 특징 추출 중... (진행률: 60%)
[오후 4:02:12] LLM 분석 중... (진행률: 80%)
[오후 4:02:13] 작업 완료! 추정 질량: 150.5g
```

#### 4. 문제 해결
- **연결 실패**: API 서버가 8001 포트에서 실행 중인지 확인
- **파일 업로드 실패**: 이미지 파일 형식 확인 (JPG, PNG)
- **진행 상황이 안 보임**: 브라우저 개발자 도구에서 WebSocket 연결 상태 확인

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

### 3단계: 특징 추출
- **목적**: 세그멘테이션과 깊이 정보 결합
- **계산 내용**:
  - 픽셀 면적 → 실제 면적 변환
  - 깊이 분포 → 부피 추정
  - 기준 물체 → 스케일 보정

### 4단계: 기준 물체 분석
- **목적**: 실제 크기 스케일 계산
- **지원 기준 물체**: 이어폰 케이스, 동전 등
- **출력**: 픽셀-센티미터 변환 비율

### 5단계: LLM 질량 추정
- **목적**: 최종 질량 계산 및 검증
- **입력**: 추출된 모든 특징
- **출력**: 그램 단위 질량, 신뢰도, 추정 근거

### 6단계: API 응답 생성
- **목적**: 클라이언트에게 결과 전송
- **형식**: JSON 응답 또는 WebSocket 실시간 알림
- **내용**: 질량 추정 결과, 신뢰도, 처리 시간 등



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
├── weights/yolo_food_v1.pt  # YOLO 모델 파일 (Git LFS)
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

#### 명령줄 실행 결과
```json
{
  "filename": "test2.jpg",
  "detected_objects": {
    "food": 1,
    "reference_objects": 1
  },
  "mass_estimation": {
    "estimated_mass_g": 150.5,
    "confidence": 0.75,
    "food_name": "김밥",
    "verification_method": "volume_based",
    "calculation_details": {
      "volume_cm3": 120.3,
      "density_g_cm3": 1.25,
      "reference_object_used": "이어폰 케이스",
      "pixel_to_cm_ratio": 0.05
    }
  }
}
```

#### API 응답 결과
```json
{
  "filename": "test2.jpg",
  "detected_objects": {
    "food": 1,
    "reference_objects": 1
  },
  "mass_estimation": {
    "estimated_mass_g": 150.5,
    "confidence": 0.75,
    "food_name": "김밥",
    "verification_method": "volume_based"
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
2. 파일 존재 확인: ls -la weights/yolo_food_v1.pt
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