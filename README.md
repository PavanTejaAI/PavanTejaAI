<p align="center">
  <img
    src="https://capsule-render.vercel.app/api?type=waving&color=0:0EA5E9,50:06B6D4,100:06B6D4&height=220&section=header&text=Pavan%20Teja%20Saikam&fontSize=44&fontColor=FFFFFF&fontAlignY=34&animation=fadeIn&desc=AI%20Engineer%20%7C%20Agentic%20AI%20%7C%20Retrieval%20%7C%20Inference%20%7C%20Systems&descAlignY=59&descSize=17"
    width="100%"
    alt="Pavan Teja Saikam"
  />
</p>

<p align="center">
  <img
    src="https://readme-typing-svg.demolab.com/?font=JetBrains+Mono&size=15&pause=1300&color=06B6D4&center=true&vCenter=true&width=920&height=45&lines=Building+production-grade+AI+systems;Agentic+AI+%7C+RAG+%7C+Vector+Search;LLM+Inference+%7C+Model+Serving;Computer+Vision+%7C+Voice+AI;On-device+AI+%7C+Efficient+Systems"
    alt="AI engineering focus"
  />
</p>

<p align="center">
  <a href="mailto:07pavanteja@gmail.com">
    <img src="https://img.shields.io/badge/Email-07pavanteja%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
  </a>
  <a href="https://www.linkedin.com/in/pavan-teja">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  <a href="https://github.com/PavanTejaAI">
    <img src="https://img.shields.io/badge/GitHub-PavanTejaAI-111827?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
</p>

<p align="center">
  <img src="https://komarev.com/ghpvc/?username=PavanTejaAI&label=Profile%20Views&color=06B6D4&style=flat-square" alt="Profile views" />
</p>

---

# About

I am an **AI Engineer with 2+ years of experience** building production systems across:

**Agentic AI · RAG · Retrieval Systems · LLM Inference · Computer Vision · Voice AI · Backend Infrastructure · Cloud Systems**

I enjoy working where **AI engineering meets systems engineering**, turning models and research ideas into software that has to perform under real production constraints.

My engineering focus is:

`Latency` · `Throughput` · `Retrieval Quality` · `Memory Efficiency` · `Reliability` · `Security` · `Observability` · `Inference Cost`

> **I build AI systems where intelligence meets engineering.**

---

# What I Build

```text
                         AI Systems
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
    Retrieval             Agents              Inference
        │                    │                    │
   Dense Search          Tool Use             GPU Serving
   Hybrid Search         MCP                   Quantization
   Graph Search          CodeAct               ONNX
   Reranking             Multi-Agent           vLLM
   RAG                   Memory                Triton
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
                       Production Layer
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
           APIs           Cloud       Observability
        FastAPI/SSE      AWS/ECS       Metrics/Logs
        WebSockets       SQS/Lambda     Reliability
```

---

# Open Source

## ⚡ swiftvec

<p align="left">
  <a href="https://github.com/PavanTejaAI/swiftvec">
    <img src="https://img.shields.io/badge/GitHub-Repository-111827?style=for-the-badge&logo=github&logoColor=white" alt="GitHub repository" />
  </a>
  <a href="https://pypi.org/project/swiftvec/">
    <img src="https://img.shields.io/badge/PyPI-Package-111827?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI package" />
  </a>
  <a href="https://crates.io/crates/swiftvec-core">
    <img src="https://img.shields.io/badge/crates.io-Rust%20Crate-374151?style=for-the-badge&logo=rust&logoColor=white" alt="Rust crate" />
  </a>
</p>

### On-device Vector Search Engine

**Rust · HNSW · SIMD · BM25 · ONNX Runtime · Quantization**

`swiftvec` is a local-first vector search engine designed around **low latency, compact memory usage and offline execution**.

> **Semantic search designed to stay close to the device.**

```mermaid
flowchart LR
    Q[Query] --> E[ONNX Embedding]
    E --> M[Matryoshka Reduction]

    M --> H[HNSW]
    H --> B[1-bit Search]

    B --> I[Int8 Candidate Retrieval]
    I --> R[Full Precision Rerank]

    Q --> BM[BM25]

    R --> F[RRF Fusion]
    BM --> F

    F --> O[Top K]
```

### Core Design

**Rust search core**

**HNSW implemented from scratch**

**Heuristic neighbor selection**

**AVX2 and FMA SIMD kernels**

**1-bit Hamming search**

**Int8 candidate retrieval**

**Full-precision reranking**

**Matryoshka 768D → 256D**

**BM25 + dense hybrid search**

**RRF fusion**

**Metadata filtering during graph traversal**

**Memory-mapped snapshots**

**Zero-copy loading**

**Quantized ONNX embeddings**

**Python package**

**Rust crate**

**Cross-platform wheels**

### Benchmark

| System       | Hardware     |         P50 |         P99 |      Recall@5 |
| :----------- | :----------- | ----------: | ----------: | ------------: |
| **swiftvec** | Intel i5     | **2.54 ms** | **3.31 ms** |     **0.960** |
| moss         | Apple M4 Pro |      3.1 ms |      5.4 ms | Not published |
| ChromaDB     | moss setup   |    351.8 ms |    538.5 ms | Not published |
| Qdrant       | moss setup   |    597.6 ms |    771.4 ms | Not published |

### Benchmark Methodology

**100K-document corpus**

**Exact brute-force oracle**

**Recall measured independently from latency**

### Python

```python
from swiftvec import SwiftVec

db = SwiftVec()

db.add_batch(
    ["doc-1", "doc-2"],
    [
        "Vector search by meaning",
        "Photosynthesis converts sunlight"
    ],
    metadatas=[
        {"topic": "ir"},
        {"topic": "bio"}
    ]
)

results = db.search(
    "how does fast similarity search work",
    top_k=3
)
```

---

## 🤖 Quantum Lens SmartSQL Agent

<p align="left">
  <a href="https://github.com/PavanTejaAI/quantum-lens-SmartSQL-Agent">
    <img src="https://img.shields.io/badge/GitHub-Repository-111827?style=for-the-badge&logo=github&logoColor=white" alt="SmartSQL Agent repository" />
  </a>
</p>

**Python · LangChain · SQLAlchemy · FastAPI**

An autonomous text-to-SQL agent that converts natural language into executable database queries.

```mermaid
flowchart LR
    Q[Natural Language] --> A[SQL Agent]

    A --> S[Schema Inspection]
    S --> G[SQL Generation]

    G --> X[SQL Execution]
    X --> V{Valid?}

    V -->|Yes| R[Structured Result]
    V -->|No| C[Self Correction]

    C --> G
```

### Capabilities

**Natural-language database querying**

**Schema inspection**

**SQL generation**

**SQL execution**

**Join correction**

**Syntax correction**

**Self-correction loops**

**SQLAlchemy integration**

**FastAPI backend**

**Structured JSON responses**

**JWT-based multi-tenancy**

---

## 🤖 AutoML Orchestrator

<p align="left">
  <a href="https://github.com/PavanTejaAI/automl-orchestrator">
    <img src="https://img.shields.io/badge/GitHub-Repository-111827?style=for-the-badge&logo=github&logoColor=white" alt="AutoML Orchestrator repository" />
  </a>
</p>

A multi-agent AutoML workflow designed to automate the path from dataset understanding to experimentation and model development.

**Dataset analysis**

**Model selection**

**Training orchestration**

**Agent-based experimentation**

**Automated ML workflows**

---

## 🗺️ Multimodal Routing Engine

<p align="left">
  <a href="https://github.com/PavanTejaAI/Multimodal-Routing-Engine">
    <img src="https://img.shields.io/badge/GitHub-Repository-111827?style=for-the-badge&logo=github&logoColor=white" alt="Multimodal Routing Engine repository" />
  </a>
</p>

**OpenStreetMap · GTFS · Graph Algorithms**

A routing engine exploring a unified representation of road and public-transit networks.

```mermaid
flowchart LR
    O[OpenStreetMap] --> R[Road Network]
    G[GTFS] --> T[Transit Network]

    R --> U[Unified Graph]
    T --> U

    U --> E[Routing Engine]
    E --> Q[Route]
```

**Graph construction**

**Road networks**

**Transit networks**

**Unified graph representation**

**Low-latency routing**

---

# Technology Stack

## Languages

<p align="left">
  <img src="https://skillicons.dev/icons?i=python,rust,typescript,javascript,nodejs,cpp&perline=6" alt="Programming languages" />
</p>

`Python` `Rust` `TypeScript` `JavaScript` `Node.js` `C++` `SQL`

---

## Generative AI

<p align="left">
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-111827?style=flat-square" />
  <img src="https://img.shields.io/badge/MCP-4B5563?style=flat-square" />
  <img src="https://img.shields.io/badge/CodeAct-374151?style=flat-square" />
  <img src="https://img.shields.io/badge/Function%20Calling-075985?style=flat-square" />
</p>

`LLMs` `RAG` `Agentic AI` `LangChain` `LangGraph` `MCP` `CodeAct` `Function Calling` `Prompt Engineering` `LoRA` `QLoRA`

---

## Retrieval and Search

<p align="left">
  <img src="https://img.shields.io/badge/Vespa-111827?style=flat-square&logo=vespa&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-0467DF?style=flat-square" />
  <img src="https://img.shields.io/badge/Neo4j-4581C3?style=flat-square&logo=neo4j&logoColor=white" />
  <img src="https://img.shields.io/badge/FalkorDB-111827?style=flat-square" />
  <img src="https://img.shields.io/badge/Weaviate-FF5C5C?style=flat-square" />
  <img src="https://img.shields.io/badge/LanceDB-111827?style=flat-square" />
</p>

`Vespa` `FalkorDB` `Neo4j` `FAISS` `Weaviate` `LanceDB` `HNSW` `Embeddings` `BM25` `Dense Retrieval` `Hybrid Search` `RRF` `Binary Quantization`

---

## Machine Learning and Deep Learning

<p align="left">
  <img src="https://skillicons.dev/icons?i=pytorch,tensorflow,sklearn&perline=3" alt="Machine learning stack" />
</p>

<p align="left">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
</p>

`PyTorch` `TensorFlow` `Scikit-learn` `Hugging Face Transformers` `NumPy` `Pandas`

---

## Inference and Model Serving

<p align="left">
  <img src="https://img.shields.io/badge/vLLM-111827?style=flat-square" />
  <img src="https://img.shields.io/badge/Triton-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/ONNX%20Runtime-005CED?style=flat-square&logo=onnx&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorRT-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/llama.cpp-111827?style=flat-square" />
  <img src="https://img.shields.io/badge/whisper.cpp-111827?style=flat-square" />
</p>

`vLLM` `Triton Inference Server` `ONNX Runtime` `TensorRT` `llama.cpp` `whisper.cpp` `FP8` `KV Cache` `Quantization` `Model Optimization`

---

## Computer Vision and Voice AI

<p align="left">
  <img src="https://skillicons.dev/icons?i=opencv&perline=1" alt="OpenCV" />
</p>

<p align="left">
  <img src="https://img.shields.io/badge/SCRFD-111827?style=flat-square" />
  <img src="https://img.shields.io/badge/MediaPipe-0097A7?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Whisper-111827?style=flat-square" />
  <img src="https://img.shields.io/badge/LiveKit-111827?style=flat-square&logo=livekit&logoColor=white" />
</p>

`SCRFD` `OpenCV` `MediaPipe` `Face Recognition` `Speech-to-Text` `Speaker Diarization` `Speaker Verification` `Speech Enhancement` `Source Separation` `LiveKit Agents`

---

## Backend and APIs

<p align="left">
  <img src="https://skillicons.dev/icons?i=fastapi,nodejs,express,redis,mongodb,mysql&perline=6" alt="Backend stack" />
</p>

`FastAPI` `REST APIs` `WebSockets` `SSE` `Redis` `MongoDB` `MySQL` `Node.js` `Express`

---

## Cloud and Infrastructure

<p align="left">
  <img src="https://skillicons.dev/icons?i=aws,gcp,docker,kubernetes,linux&perline=5" alt="Cloud and infrastructure" />
</p>

<p align="left">
  <img src="https://img.shields.io/badge/ECS-FF9900?style=flat-square&logo=amazonecs&logoColor=white" />
  <img src="https://img.shields.io/badge/Lambda-FF9900?style=flat-square&logo=awslambda&logoColor=white" />
  <img src="https://img.shields.io/badge/S3-569A31?style=flat-square&logo=amazons3&logoColor=white" />
  <img src="https://img.shields.io/badge/SQS-FF4F8B?style=flat-square&logo=amazonsqs&logoColor=white" />
  <img src="https://img.shields.io/badge/ECR-FF9900?style=flat-square&logo=amazonaws&logoColor=white" />
</p>

`AWS` `GCP` `ECS` `EC2` `Lambda` `SQS` `S3` `ECR` `Vertex AI` `Docker` `Kubernetes` `Linux` `CI/CD`

---

## Observability and Engineering

<p align="left">
  <img src="https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=prometheus&logoColor=white" />
  <img src="https://img.shields.io/badge/Grafana-F46800?style=flat-square&logo=grafana&logoColor=white" />
  <img src="https://img.shields.io/badge/Loki-111827?style=flat-square&logo=grafana&logoColor=white" />
  <img src="https://img.shields.io/badge/Git-181717?style=flat-square&logo=git&logoColor=white" />
</p>

`Prometheus` `Grafana` `Loki` `Git` `CI/CD` `Containerized Deployments` `Monitoring` `Production Operations`

---

# Engineering Principles

### Measure First

**Latency · Throughput · Recall · Accuracy · Memory · GPU Utilization · Cost**

### Optimize the Bottleneck

**Quantization · Batching · Caching · Reranking · SIMD · Efficient Indexing · Local Inference**

### Retrieval Is a System

```text
Ingestion
   ↓
Chunking
   ↓
Indexing
   ↓
Retrieval
   ↓
Filtering
   ↓
Reranking
   ↓
Fusion
   ↓
Evaluation
```

### Production Means Failure Handling

**Timeouts · Retries · Fallbacks · Isolation · Security · Observability · Rollbacks · Cost Controls**

### Reproducibility Matters

**Benchmarks over opinions**

**Evaluation over demos**

**Measured trade-offs over hype**

---

# Experience

| Organization                      | Role                      | Period              |
| :-------------------------------- | :------------------------ | :------------------ |
| **VE The Intent Company**         | AI Engineer               | Aug 2024 - Present  |
| **Sankalpa Projects Consultancy** | Software Developer Intern | Jul 2023 - Jan 2024 |
| **Ranvi Technologies**            | Software Developer Intern | Feb 2023 - Jul 2023 |

---

# Education

## Malla Reddy College of Engineering & Technology

**B.Tech, Computer Science & Engineering, Data Science**

2021 - 2024 · Hyderabad, India

`Machine Learning` `Deep Learning` `NLP` `Data Mining` `Big Data Analytics` `DBMS` `Data Structures & Algorithms`

## Madhira Institute of Technology & Sciences

**Diploma, Mechanical Engineering**

2018 - 2021

---

# Current Focus

### Agentic AI

`MCP` `Tool Use` `CodeAct` `Multi-Agent Systems` `Planning` `Orchestration` `Long-term Memory`

### Retrieval

`RAG` `Vector Search` `Graph Retrieval` `Hybrid Search` `Quantization` `Reranking` `Retrieval Evaluation`

### Inference

`GPU Serving` `Quantization` `Batching` `KV Cache Optimization` `Low-latency Inference`

### On-device AI

`Local LLMs` `Private Inference` `On-device Retrieval` `SIMD` `Efficient Local Compute`

---

# The Problem I Like Working On

```text
                 AI Applications
                        │
                        ▼
                 More Intelligent
                        │
                        ▼
                    More Useful
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
        Faster        Cheaper       Private
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                    Reliable
                        │
                        ▼
                  Production Scale
```

> **How do we make AI systems faster, cheaper, more reliable, more accurate and more private while keeping them practical to operate?**

That is the class of problems I want to keep solving.

---

# GitHub Activity

<p align="left">
  <img
    src="https://github-readme-stats.vercel.app/api?username=PavanTejaAI&show_icons=true&theme=default&title_color=06B6D4&icon_color=06B6D4&text_color=E2E8F0&bg_color=00000000&hide_border=true"
    height="170"
    alt="GitHub statistics"
  />
</p>

<p align="left">
  <img
    src="https://streak-stats.demolab.com?user=PavanTejaAI&theme=default&background=00000000&stroke=0EA5E9&ring=06B6D4&fire=0EA5E9&currStreakLabel=0EA5E9&sideLabels=E2E8F0&currStreakNum=FFFFFF&dates=E2E8F0&hide_border=true"
    height="170"
    alt="GitHub streak"
  />
</p>

### Contribution Graph

<p align="left">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="https://raw.githubusercontent.com/PavanTejaAI/PavanTejaAI/output/snake-dark.svg"
    />
    <source
      media="(prefers-color-scheme: light)"
      srcset="https://raw.githubusercontent.com/PavanTejaAI/PavanTejaAI/output/snake.svg"
    />
    <img
      src="https://raw.githubusercontent.com/PavanTejaAI/PavanTejaAI/output/snake.svg"
      alt="GitHub contribution snake"
    />
  </picture>
</p>

---

# Connect

<p align="left">
  <a href="mailto:07pavanteja@gmail.com">
    <img src="https://img.shields.io/badge/Email-07pavanteja%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
  </a>
  <a href="https://www.linkedin.com/in/pavan-teja">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  <a href="https://github.com/PavanTejaAI">
    <img src="https://img.shields.io/badge/GitHub-Explore%20My%20Work-111827?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
</p>

<p align="left">
  <strong>Building AI systems where intelligence meets engineering.</strong>
</p>

<p align="center">
  <img
    src="https://capsule-render.vercel.app/api?type=waving&color=0:0EA5E9,50:06B6D4,100:06B6D4&height=120&section=footer"
    width="100%"
    alt=""
  />
</p>
