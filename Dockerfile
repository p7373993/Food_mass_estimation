# Python 3.12 slim 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# pyproject.toml과 잠금 파일 복사
COPY pyproject.toml .

# uv.lock 파일이 있다면 아래 줄의 주석을 해제하세요.
# COPY uv.lock .

# uv 설치
RUN pip install uv

# pyproject.toml 기반 의존성 설치 (dev 제외)
RUN uv pip install --no-cache --system .

# 만약 dev까지 설치하고 싶으면
# RUN uv pip install --no-cache --system --group dev .

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p logs results data weights

# 포트 노출
EXPOSE 8001

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 애플리케이션 실행
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]