# GCP-multi-cluster-memory-bank

## Overview
이 프로젝트는 Vertex AI Vector Search (KNN), PySpark, AnolLM을 활용하여 대규모 데이터 환경에서 이상(Anomaly)을 효과적으로 탐지하는 확장성 높은 백엔드 시스템을 구축하는 것을 목표로 합니다.

특히, 금융 거래 데이터와 같은 대규모 정형 데이터에 대해 최신 임베딩 기술(AnolLM)을 적용하고, 이를 Google Cloud의 완전 관리형 서비스들과 연동하여 모델 개발부터 대규모 데이터 추론 및 모니터링까지 전 과정을 자동화하는 MLOps 파이프라인을 구현합니다.

## 아키텍처 다이어그램 (Architecture Diagram)
[아키텍처 이미지 삽입]

예시 다이어그램의 흐름:
BigQuery → (PySpark 분산 처리) → Dataproc Serverless → (AnolLM 임베딩 생성) → BigQuery → (Vector Search 인덱싱) → Vertex AI Vector Search → (Feature Monitoring)

## 핵심 기술 스택 (Core Technology Stack)
- 데이터 처리: PySpark on Dataproc Serverless (BigQuery Spark)
- 데이터 스토리지: Google BigQuery, Google Cloud Storage
- 임베딩 모델: AnolLM (Transformer 기반 Tabular Anomaly Embedding)
- 이상 탐지: Vertex AI Vector Search (KNN 기반 고속 유사도 검색)
- MLOps & 자동화: Vertex AI Pipelines, Vertex AI Feature Store
- 프로그래밍 언어: Python 3.9+
- 주요 라이브러리: transformers, torch, pyspark, pandas, google-cloud-aiplatform

## 프로젝트 실행 단계 (Project Execution Steps)
### Google Cloud 환경 설정
1. 필수 API 활성화:
BigQuery API, Dataproc API, Cloud Storage API 등

2. IAM 권한 설정:
Dataproc 서비스 계정(991692518087-compute@developer.gserviceaccount.com)에 Storage 관리자 및 BigQuery 데이터 편집자 권한 부여.

### 데이터 및 모델 준비
1. 원시 데이터: BigQuery에 anollm.fraud_data_original.train_raw_query 테이블로 데이터 로드.
2. AnolLM 모델: 사전 학습된 AnolLM 모델 파일을 gs://your-model-bucket/anollm-model/ 경로에 업로드.
3. Spark 코드: main.py와 requirements.txt 파일을 gs://your-code-bucket/ 경로에 업로드.

### AnolLM 임베딩 추론 (PySpark)
다음 gcloud 명령어를 사용하여 Dataproc Serverless Batch Job을 제출합니다. 이 잡은 BigQuery에서 데이터를 읽고, AnolLM 임베딩을 생성한 후 결과를 새로운 BigQuery 테이블에 저장합니다.
````
gcloud dataproc batches submit spark \
    --project="your-gcp-project-id" \
    --region="your-gcp-region" \
    --batch="anol-lm-embedding-batch-$(date +%Y%m%d%H%M%S)" \
    --py-files="gs://your-code-bucket/main.py" \
    ... (전체 명령어는 첨부된 `run_job.sh` 스크립트 참고)
````

### 이상 탐지 시스템 연동
1. Vertex AI Feature Store: PySpark 결과 테이블을 Feature Group으로 등록하여 임베딩 벡터를 특징으로 관리.
2. Vertex AI Vector Search: 생성된 임베딩 벡터를 Vector Search 인덱스로 빌드하여 고속 KNN 검색을 위한 기반 마련.
3. Feature Monitoring: Vertex AI를 통해 임베딩 데이터의 드리프트를 주기적으로 모니터링하여 데이터 품질 이상 감지.

## 결과 및 성과 (Results & Achievements)
- 대규모 데이터 처리: 약 1.0GB 규모의 AnolLM 모델을 Spark 환경에 통합하여 수억 건의 데이터에 대한 임베딩 추론을 성공적으로 분산 처리.
- 비용 효율성: Dataproc Serverless를 활용하여 클러스터 관리 및 유휴 비용 없이 필요한 시점에만 컴퓨팅 자원을 사용.
- 종합적인 MLOps 역량: 데이터 전처리(PySpark) → 임베딩 추론(AnolLM) → 특징 관리(Feature Store) → 이상 탐지(Vector Search) → 모니터링(Feature Monitor)에 이르는 End-to-End ML 파이프라인을 구축.
- 핵심 코드: main.py 파일에서 대규모 분산 환경에서 AnolLM 모델을 활용한 임베딩 추론 로직을 확인할 수 있습니다.

## 참조 (Citations)
[AnolLM](https://github.com/amazon-science/AnoLLM-large-language-models-for-tabular-anomaly-detection)\
[Google Cloud Dataproc Serverless](https://cloud.google.com/dataproc-serverless/docs/overview?hl=ko)\
[Vertex AI Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview?hl=ko)
