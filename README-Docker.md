# 🐳 ML 서버 Docker 배포 가이드

## 📋 목차
- [개요](#개요)
- [사전 요구사항](#사전-요구사항)
- [빠른 시작](#빠른-시작)
- [환경 설정](#환경-설정)
- [배포 방법](#배포-방법)
- [모니터링](#모니터링)
- [문제 해결](#문제-해결)

## 🎯 개요

ML 서버를 Docker 컨테이너로 배포하여 환경 의존성 없이 어디서든 실행할 수 있습니다.

### 지원 환경
- **개발 환경**: 코드 변경사항 실시간 반영
- **프로덕션 환경**: 최적화된 성능과 안정성
- **Nginx 프록시**: 로드 밸런싱 및 SSL 지원

## 🔧 사전 요구사항

### 필수 소프트웨어
```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 시스템 요구사항
- **CPU**: 최소 2코어 (권장 4코어)
- **메모리**: 최소 4GB (권장 8GB)
- **저장공간**: 최소 10GB
- **네트워크**: 인터넷 연결 (API 키 사용)

## 🚀 빠른 시작

### 1. 환경 변수 설정
```bash
# .env 파일 생성
cat > .env << EOF
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
DEBUG_MODE=false
ENABLE_MULTIMODAL=true
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-2.5-flash
EOF
```

### 2. Docker 이미지 빌드 및 실행
```bash
# 스크립트 사용 (권장)
./scripts/docker-build.sh build
./scripts/docker-build.sh prod

# 또는 직접 실행
docker-compose up -d
```

### 3. 서비스 확인
```bash
# 상태 확인
./scripts/docker-build.sh status

# 로그 확인
./scripts/docker-build.sh logs
```

## ⚙️ 환경 설정

### 환경 변수 설명

| 변수명 | 설명 | 기본값 | 필수 |
|--------|------|--------|------|
| `GEMINI_API_KEY` | Google Gemini API 키 | - | ✅ |
| `OPENAI_API_KEY` | OpenAI API 키 | - | ❌ |
| `DEBUG_MODE` | 디버그 모드 활성화 | false | ❌ |
| `ENABLE_MULTIMODAL` | 멀티모달 기능 활성화 | true | ❌ |
| `LLM_PROVIDER` | LLM 제공자 (gemini/openai) | gemini | ❌ |
| `LLM_MODEL_NAME` | LLM 모델명 | gemini-2.5-flash | ❌ |

### .env 파일 예시
```bash
# API 키 설정
GEMINI_API_KEY=AIzaSyC...
OPENAI_API_KEY=sk-...

# 기능 설정
DEBUG_MODE=false
ENABLE_MULTIMODAL=true
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-2.5-flash

# 로깅 설정
LOG_LEVEL=INFO
```

## 🚀 배포 방법

### 1. 개발 환경 배포
```bash
# 개발 환경 실행 (포트 8002)
./scripts/docker-build.sh dev

# 접속 URL
# API 서버: http://localhost:8002
# API 문서: http://localhost:8002/docs
```

### 2. 프로덕션 환경 배포
```bash
# 프로덕션 환경 실행 (포트 8001)
./scripts/docker-build.sh prod

# 접속 URL
# API 서버: http://localhost:8001
# API 문서: http://localhost:8001/docs
```

### 3. Nginx와 함께 배포
```bash
# Nginx 프록시와 함께 실행 (포트 80)
./scripts/docker-build.sh nginx

# 접속 URL
# API 서버: http://localhost
# API 문서: http://localhost/docs
```

### 4. 수동 배포
```bash
# 이미지 빌드
docker build -t food-calorie-ml-server:latest .

# 개발 환경
docker-compose --profile dev up -d

# 프로덕션 환경
docker-compose -f docker-compose.prod.yml up -d

# Nginx 포함
docker-compose -f docker-compose.prod.yml --profile nginx up -d
```

## 📊 모니터링

### 컨테이너 상태 확인
```bash
# 모든 컨테이너 상태
docker-compose ps

# 특정 서비스 로그
docker-compose logs ml-server

# 실시간 로그 확인
docker-compose logs -f ml-server
```

### 헬스체크
```bash
# API 헬스체크
curl http://localhost:8001/health

# 컨테이너 헬스체크
docker inspect food-calorie-ml-server | grep Health -A 10
```

### 리소스 사용량 확인
```bash
# 컨테이너 리소스 사용량
docker stats food-calorie-ml-server

# 디스크 사용량
docker system df
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 포트 충돌
```bash
# 포트 사용 확인
netstat -tulpn | grep :8001

# 다른 포트 사용
docker-compose up -d -p 8003:8001
```

#### 2. 메모리 부족
```bash
# 메모리 제한 설정
docker-compose -f docker-compose.prod.yml up -d
# (docker-compose.prod.yml에 메모리 제한이 설정되어 있음)
```

#### 3. API 키 오류
```bash
# 환경 변수 확인
docker-compose exec ml-server env | grep API_KEY

# .env 파일 재생성
./scripts/docker-build.sh
```

#### 4. 모델 파일 누락
```bash
# weights 디렉토리 확인
ls -la weights/

# 모델 파일 다운로드 (필요시)
# weights/yolo_food_v1.pt 파일이 필요합니다
```

### 로그 분석
```bash
# 에러 로그만 확인
docker-compose logs ml-server | grep ERROR

# 최근 로그 확인
docker-compose logs --tail=100 ml-server

# 특정 시간대 로그
docker-compose logs --since="2025-01-01T00:00:00" ml-server
```

### 컨테이너 재시작
```bash
# 서비스 재시작
./scripts/docker-build.sh restart

# 또는 수동 재시작
docker-compose restart ml-server
```

## 🔄 업데이트 및 유지보수

### 이미지 업데이트
```bash
# 최신 코드로 이미지 재빌드
./scripts/docker-build.sh build

# 서비스 재시작
./scripts/docker-build.sh restart
```

### 데이터 백업
```bash
# 결과 파일 백업
docker cp food-calorie-ml-server:/app/results ./backup/results

# 로그 파일 백업
docker cp food-calorie-ml-server:/app/logs ./backup/logs
```

### 정리 작업
```bash
# 사용하지 않는 이미지 정리
docker image prune -f

# 사용하지 않는 볼륨 정리
docker volume prune -f

# 전체 시스템 정리
docker system prune -a
```

## 📝 스크립트 사용법

### 사용 가능한 명령어
```bash
./scripts/docker-build.sh build    # 이미지 빌드
./scripts/docker-build.sh dev      # 개발 환경 실행
./scripts/docker-build.sh prod     # 프로덕션 환경 실행
./scripts/docker-build.sh nginx    # Nginx와 함께 실행
./scripts/docker-build.sh stop     # 서비스 중지
./scripts/docker-build.sh restart  # 서비스 재시작
./scripts/docker-build.sh logs     # 로그 확인
./scripts/docker-build.sh status   # 상태 확인
./scripts/docker-build.sh help     # 도움말
```

### 스크립트 예시
```bash
# 전체 배포 과정
./scripts/docker-build.sh build    # 1. 이미지 빌드
./scripts/docker-build.sh prod     # 2. 프로덕션 실행
./scripts/docker-build.sh status   # 3. 상태 확인

# 개발 환경에서 테스트
./scripts/docker-build.sh dev      # 개발 환경 실행
curl http://localhost:8002/health  # 헬스체크
./scripts/docker-build.sh stop     # 서비스 중지
```

## 🌐 네트워크 설정

### 포트 매핑
- **개발 환경**: 8002 → 8001
- **프로덕션 환경**: 8001 → 8001
- **Nginx 프록시**: 80 → 8001

### 방화벽 설정
```bash
# Ubuntu/Debian
sudo ufw allow 8001
sudo ufw allow 80

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8001/tcp
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --reload
```

## 🔒 보안 고려사항

### 환경 변수 보안
```bash
# .env 파일 권한 설정
chmod 600 .env

# 프로덕션에서는 시크릿 관리 사용
docker secret create gemini_api_key .env
```

### 네트워크 보안
```bash
# 내부 네트워크만 사용
docker network create ml-internal
docker-compose --network ml-internal up -d
```

### 컨테이너 보안
```bash
# 비루트 사용자로 실행
docker run --user 1000:1000 food-calorie-ml-server:latest

# 리소스 제한
docker run --memory=2g --cpus=1.0 food-calorie-ml-server:latest
```

---

## 📞 지원

문제가 발생하면 다음을 확인해주세요:

1. **로그 확인**: `./scripts/docker-build.sh logs`
2. **상태 확인**: `./scripts/docker-build.sh status`
3. **환경 변수 확인**: `.env` 파일 설정
4. **시스템 리소스**: 메모리, CPU 사용량

더 자세한 정보는 [메인 README.md](README.md)를 참조하세요. 