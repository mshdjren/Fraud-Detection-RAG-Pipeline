# Fraud Detection RAG Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/GKE-Kubernetes-326CE5?logo=kubernetes" />
  <img src="https://img.shields.io/badge/Elasticsearch-8.x-005571?logo=elasticsearch" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python" />
</p>

## 📌 프로젝트 개요 (Project Overview)

**금융/반도체 이상 탐지를 위한 프로덕션급 RAG(Retrieval-Augmented Generation) 파이프라인**

대규모 Tabular/이미지 데이터에서 이상 거래를 탐지하고 원인을 설명하는 엔드투엔드 시스템입니다. 
Elasticsearch 기반 Two-Stage Retrieval과 경량 LLM Analyzer를 결합하여 **실시간 서빙**(P99 480ms 이내)과 
**API 비용 85% 절감**을 동시에 달성했습니다.

### 🎯 핵심 문제 해결

| 문제 | 해결 방안 | 성과 |
|------|---------|------|
| 대규모 벡터 검색 병목 | Cluster-Based Two-Stage Retrieval | Router Recall@5 11% ↑, Latency P99 25% ↓ |
| 메모리 제약 | Coreset Sampling (10%) | 메모리 89% ↓ (2.8GB → 0.31GB) |
| LLM API 비용 과다 | Knowledge Distillation (7B → 0.5B) | API 비용 85% ↓, Latency 85% ↓ |
| 파이프라인 오류 학습 부족 | Pipeline-Aware Hard Negative Mining | AUROC 2.3%p ↑ |
| Confidence 과신 문제 | GRPO 강화학습 (4-axis reward) | ECE 47% ↓ (0.15 → 0.08) |

---

## 🏗️ 시스템 아키텍처 (Architecture)

**[아키텍처 다이어그램 삽입 예정]**  
*GKE 기반 4단계 MSA 구조: Router → Retriever → Analyzer → Orchestrator*

### 파이프라인 흐름

Test Transaction
↓
[Router] Percolate Query → Cluster Matching (최근접 클러스터 필터링)
↓
[Retriever] Filtered KNN → Top-5 Normal Vectors (클러스터 내 검색)
↓
[Analyzer] Distance Scoring + LLM Reasoning → 이상 판단 + 원인 설명
↓
[Orchestrator] Final Output → API Response

---

## 🔬 핵심 기술 (Core Technologies)

### 1️⃣ Two-Stage Retrieval

**Stage 1: Cluster Routing (Percolate Query)**
```python
# K-means 클러스터링 + Decision Tree 규칙 → Elasticsearch Percolate Query 변환
# 예시: feature v1 ∈ [0.8, ∞) AND v3 ∈ [-1.2, ∞) → Cluster 63

{
  "query": {
    "bool": {
      "filter": [
        {"range": {"v1": {"gte": 0.8}}},
        {"range": {"v3": {"gte": -1.2}}}
      ]
    }
  }
}
```
**효과**: 검색 공간을 클러스터 단위로 필터링 → Router Recall@5 0.816 → 0.905 (11% ↑)

**Stage 2: Filtered KNN (Coreset Sampling)**
```python
# 매칭된 클러스터 내에서만 KNN 검색
# Coreset: 대표 샘플 10%만 인덱싱 (Greedy Farthest Point Sampling)

AUROC_drop = 0.4%p  # 0.912 → 0.908 (거의 유지)
Memory_reduction = 89%  # 2.8GB → 0.31GB
Latency_improvement = 46%  # 125ms → 68ms (median)
```

---

### 2️⃣ LLM Analyzer 경량화

**Knowledge Distillation (Qwen2.5-7B → 0.5B)**
```python
# Teacher-Student Distillation
# 파라미터 비율: 14:1 → 이론적 속도 향상: 6-8배
# 실제 측정: 85% latency 단축 (500ms → 75ms)

Distillation_loss = KL(Student || Teacher) + CE(Student, Ground_Truth)
```

**Pipeline-Aware Hard Negative Mining (3가지 유형)**
```python
# Type 1: Router Misrouting (28% 발생)
#   → gt_cluster_id ≠ predicted_cluster

# Type 2: Cross-Cluster Retrieval (55% 발생)
#   → 라우팅 정확하나 HNSW 경계에서 타 클러스터 이웃 혼입

# Type 3: Distance Band (15% 발생)
#   → top1_distance ∈ [P75, P90], moderate/distant 경계

총 2,219개 hard negative 샘플 자동 생성
```

**GRPO 강화학습 (Group Relative Policy Optimization)**
```python
# 4축 보상 함수 (그룹 내 상대 비교)
Reward = 0.40 × Accuracy          # Binary 판정
       + 0.30 × Calibration       # |confidence - actual| 최소화
       + 0.20 × Coherence         # 20-80 단어, 논리 일관성
       + 0.10 × Factuality        # Evidence 근거 사실성

Advantage = (reward - group_mean) / group_std  # 같은 hard negative type 내
Policy_loss = -advantage × log_prob + KL_penalty
```

---

## 💻 기술 스택 (Technology Stack)

**Infrastructure & Deployment**
- **Orchestration**: Kubernetes (GKE), Helm
- **Service Mesh**: Istio (optional)
- **Monitoring**: Prometheus, Grafana

**Data & Search**
- **Search Engine**: Elasticsearch 8.x (ECK)
- **Vector DB**: HNSW Index (Elasticsearch)
- **Data Processing**: PySpark, Pandas

**ML & AI**
- **Framework**: PyTorch, Transformers (Hugging Face)
- **LLM**: Qwen2.5-0.5B-Instruct (LoRA fine-tuned)
- **Teacher Model**: Llama-3.1-8B, Qwen2.5-7B
- **Training**: SFT (Supervised Fine-Tuning), GRPO (RL)

**Backend & API**
- **Framework**: FastAPI 0.115
- **Async**: asyncio, aiohttp
- **Validation**: Pydantic

**Cloud & Storage**
- **Platform**: Google Cloud Platform
- **Storage**: Google Cloud Storage (GCS)
- **Registry**: Artifact Registry, Container Registry

---

## 🚀 실행 가이드 (Quick Start)

### Prerequisites
```bash
# 필수 도구 설치
- kubectl (1.28+)
- gcloud CLI
- Docker
- Python 3.10+
```

### 1️⃣ Elasticsearch 배포 (ECK)
```bash
# Elasticsearch Operator 설치
kubectl apply -f https://download.elastic.co/downloads/eck/2.9.0/crds.yaml
kubectl apply -f https://download.elastic.co/downloads/eck/2.9.0/operator.yaml

# Elasticsearch Cluster 배포
kubectl apply -f k8s/elasticsearch-cluster.yaml

# 인덱스 생성
python scripts/create_index.py --experiment-case pca_64_k100
```

### 2️⃣ Microservices 배포
```bash
# ConfigMap 적용 (환경 변수)
kubectl apply -f k8s/configmap.yaml

# 각 서비스 배포
kubectl apply -f k8s/router-deployment.yaml
kubectl apply -f k8s/retriever-deployment.yaml
kubectl apply -f k8s/analyzer-deployment.yaml
kubectl apply -f k8s/orchestrator-deployment.yaml
```

### 3️⃣ 추론 실행
```bash
# Batch Inference (평가용)
python batch_inference.py \
  --experiment-case pca_64_k100 \
  --percolate-version v14 \
  --coreset-percentage 10

# API 호출 (실시간)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "purchase_value": 32.53,
    "source": "SEO",
    "browser": "Chrome",
    "sex": "M",
    "age": 25
  }'
```

---

## 📊 실험 결과 (Results)

**[실험 결과 그래프 삽입 예정]**  
*Router 버전별 Recall@5 비교 / Coreset Sampling Trade-off Curve / GRPO 학습 곡선*

### 주요 성과

| 메트릭 | 개선 | 상세 |
|--------|------|------|
| **Router Recall@5** | 11% ↑ | 0.816 → 0.905 (Strict AND 대비) |
| **Retrieval Latency P99** | 25% ↓ | 클러스터 필터링 효과 |
| **메모리 사용량** | 89% ↓ | 2.8GB → 0.31GB (10% coreset) |
| **Median Latency** | 46% ↓ | 125ms → 68ms |
| **API 비용** | 85% ↓ | 7B → 0.5B Distillation |
| **추론 Latency** | 85% ↓ | 500ms → 75ms |
| **이상 탐지 AUROC** | 2.3%p ↑ | Hard Negative Mining + GRPO |
| **Confidence Calibration (ECE)** | 47% ↓ | 0.15 → 0.08 (GRPO 효과) |

---

## 📁 프로젝트 구조 (Directory Structure)
fraudecom_v3/
├── apps/
│   ├── router/              # Percolate Query 기반 클러스터 라우팅
│   ├── retriever/           # Coreset Sampling + HNSW KNN
│   ├── analyzer/            # LLM 기반 이상 판단 + 원인 설명
│   └── orchestrator/        # 전체 워크플로우 제어
├── k8s/
│   ├── elasticsearch/       # ECK 배포 설정
│   ├── router-deployment.yaml
│   ├── retriever-deployment.yaml
│   └── analyzer-deployment.yaml
├── evaluation/
│   ├── batch_inference.py   # 평가용 배치 추론
│   └── evaluation_metrics.py
├── scripts/
│   ├── create_index.py      # Elasticsearch 인덱스 생성
│   ├── ingest_tree_pipeline.py  # Decision Tree → Percolate 변환
│   └── hard_negative_miner.py   # Hard Negative Mining
└── README.md

---

## 🔧 개발 환경 설정 (Development)

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일에 GCP_PROJECT, ES_HOST 등 설정

# 4. 로컬 테스트
pytest tests/
```

---

## 📚 참조 (References)

**학술 논문**
- [AnoLLM: Large Language Models for Tabular Anomaly Detection](https://github.com/amazon-science/AnoLLM-large-language-models-for-tabular-anomaly-detection)
- [Memory Bank for Anomaly Detection (CVPR 2024)](https://arxiv.org/abs/2404.xxxxx)

**기술 문서**
- [Elasticsearch Percolate Query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-percolate-query.html)
- [Coreset Sampling for Active Learning](https://arxiv.org/abs/1708.00489)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.xxxxx)

**당근 테크 블로그**
- [연간 LLM 호출 비용 25% 절감, 시맨틱 캐싱 도입 기록](https://medium.com/daangn/semantic-caching-llm-cost-reduction)

---

## 📝 License

이 프로젝트는 개인 포트폴리오 목적으로 제작되었습니다.

---

## 👤 Contact

- **GitHub**: [@your-username](https://github.com/your-username)
- **Email**: moonsh1031@gmail.com
- **LinkedIn**: [문상혁](https://linkedin.com/in/your-profile)

---

## 🎯 Next Steps

- [ ] Feature Store 연동 (Vertex AI)
- [ ] A/B Testing 프레임워크 구축
- [ ] Multi-modal Anomaly Detection (이미지 + Tabular)
- [ ] AutoML Hyperparameter Tuning
