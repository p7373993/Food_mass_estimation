#!/bin/bash

# ML 서버 Docker 빌드 및 실행 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 환경 변수 파일 확인
check_env_file() {
    if [ ! -f .env ]; then
        log_warning ".env 파일이 없습니다. 생성합니다..."
        cat > .env << EOF
# ML 서버 환경 변수
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
DEBUG_MODE=false
ENABLE_MULTIMODAL=true
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-2.5-flash
EOF
        log_info ".env 파일이 생성되었습니다. API 키를 설정해주세요."
        exit 1
    fi
}

# Docker 이미지 빌드
build_image() {
    log_info "Docker 이미지를 빌드합니다..."
    docker build -t food-calorie-ml-server:latest .
    log_success "Docker 이미지 빌드 완료"
}

# 개발 환경 실행
run_dev() {
    log_info "개발 환경을 시작합니다..."
    docker-compose --profile dev up -d
    log_success "개발 환경이 시작되었습니다."
    log_info "API 서버: http://localhost:8002"
    log_info "API 문서: http://localhost:8002/docs"
}

# 프로덕션 환경 실행
run_prod() {
    log_info "프로덕션 환경을 시작합니다..."
    docker-compose -f docker-compose.prod.yml up -d
    log_success "프로덕션 환경이 시작되었습니다."
    log_info "API 서버: http://localhost:8001"
    log_info "API 문서: http://localhost:8001/docs"
}

# Nginx와 함께 실행
run_with_nginx() {
    log_info "Nginx와 함께 프로덕션 환경을 시작합니다..."
    docker-compose -f docker-compose.prod.yml --profile nginx up -d
    log_success "Nginx와 함께 프로덕션 환경이 시작되었습니다."
    log_info "API 서버: http://localhost"
    log_info "API 문서: http://localhost/docs"
}

# 서비스 중지
stop_services() {
    log_info "서비스를 중지합니다..."
    docker-compose down
    docker-compose -f docker-compose.prod.yml down
    log_success "서비스가 중지되었습니다."
}

# 로그 확인
show_logs() {
    log_info "ML 서버 로그를 확인합니다..."
    docker-compose logs -f ml-server
}

# 컨테이너 상태 확인
check_status() {
    log_info "컨테이너 상태를 확인합니다..."
    docker-compose ps
    echo ""
    log_info "헬스체크 결과:"
    curl -f http://localhost:8001/health || log_error "서버가 응답하지 않습니다."
}

# 컨테이너 재시작
restart_services() {
    log_info "서비스를 재시작합니다..."
    docker-compose restart
    log_success "서비스가 재시작되었습니다."
}

# 컨테이너 직접 실행(run)
run_container() {
    log_info "기존 이미지를 사용해 컨테이너를 실행합니다..."
    docker run -d --name food-calorie-ml-server -p 8001:8001 --env-file .env food-calorie-ml-server:latest
    log_success "컨테이너가 실행되었습니다. (http://localhost:8001)"
}

# 도움말
show_help() {
    echo "ML 서버 Docker 관리 스크립트"
    echo ""
    echo "사용법: $0 [명령어]"
    echo ""
    echo "명령어:"
    echo "  build     - Docker 이미지 빌드"
    echo "  run       - 빌드 없이 기존 이미지를 컨테이너로 실행"
    echo "  dev       - 개발 환경 실행 (포트 8002)"
    echo "  prod      - 프로덕션 환경 실행 (포트 8001)"
    echo "  nginx     - Nginx와 함께 프로덕션 환경 실행 (포트 80)"
    echo "  stop      - 모든 서비스 중지"
    echo "  restart   - 서비스 재시작"
    echo "  logs      - 로그 확인"
    echo "  status    - 컨테이너 상태 확인"
    echo "  help      - 이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  $0 build    # 이미지 빌드"
    echo "  $0 run      # 기존 이미지로 컨테이너 실행"
    echo "  $0 dev      # 개발 환경 실행"
    echo "  $0 prod     # 프로덕션 환경 실행"
}

# 메인 로직
main() {
    case "${1:-help}" in
        build)
            check_env_file
            build_image
            ;;
        run)
            check_env_file
            run_container
            ;;
        dev)
            check_env_file
            build_image
            run_dev
            ;;
        prod)
            check_env_file
            build_image
            run_prod
            ;;
        nginx)
            check_env_file
            build_image
            run_with_nginx
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs
            ;;
        status)
            check_status
            ;;
        help|*)
            show_help
            ;;
    esac
}

# 스크립트 실행
main "$@" 