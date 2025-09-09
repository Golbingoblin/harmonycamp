# 1) 가벼운 Python 베이스 이미지
FROM python:3.11-slim

# 2) 시스템 패키지 (rtmidi 빌드/런타임에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libasound2-dev \
  && rm -rf /var/lib/apt/lists/*

# 3) 작업 디렉토리
WORKDIR /app

# 4) 파이썬 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) 앱 소스 복사
COPY app.py harmonizator_keyed.py chord.CSV prog16.CSV ./
COPY accomp_patterns ./accomp_patterns
COPY static ./static

# 6) 컨테이너 포트
EXPOSE 8000

# 7) 실행 커맨드 (Render가 $PORT를 내려줄 때도 대응)
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
