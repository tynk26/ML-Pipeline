# 🚗 자율주행 데이터 통합 및 검색 API 시스템 (ML Data SQL API)

본 프로젝트는 자율주행 차량에서 수집된 센서 데이터(odds.csv), 객체 인식 결과(labels.csv), 그리고 원본 선택 (selections.json) 데이터를 통합하여 정제된 학습용 데이터셋을 구축하고 관리하는 RESTful API 서버입니다.

---

# 📋 1. 프로젝트 개요

자율주행 데이터는 날씨, 온도 등 정형 데이터와 객체 카운트와 같은 비정형 데이터가 혼재되어 있습니다.  
본 시스템은 데이터 파이프라인(analyze)을 통해 오류 데이터를 자동으로 분류(Rejection)하고, 통합된 데이터를 사용자가 원하는 정밀한 조건(Sandwich Search)으로 검색할 수 있게 설계되었습니다.

---

# 🧠 2. 설계 및 데이터 파이프라인

### 2.1 데이터 처리 흐름

Raw Data
→ ODD Tagging (Stage 1)
→ Auto Labeling Validation (Stage 2)
→ Integrated Data (정상)
→ Rejections (오류 데이터)

본 시스템은 데이터의 신뢰성을 보장하기 위해 **순차적 검증 파이프라인(Sequential Validation Pipeline)**을 채택하였습니다.  
이는 상위 단계의 데이터가 무결하지 않을 경우 하위 단계의 검증을 생략하고 즉시 격리하는 아키텍처입니다.

#### 순차적 검증 로직 (Sequential Validation Logic)

- 계층적 데이터 무결성 (ODD 우선 원칙)  
  자율주행 도메인에서 **주행 환경 정보(ODD)**는 객체 라벨링 데이터의 전제 조건입니다.  
  기상, 시간, 노면 상태 등 컨텍스트가 결여된 라벨 데이터는 모델 학습용으로 부적합하므로, Stage 1 (ODD Tagging) 검증을 최우선 게이트웨이로 배치했습니다.

- 연산 효율성 및 처리 최적화  
  순차적 접근 방식을 통해 1단계에서 결함이 발견된 레코드는 2단계(Auto Labeling) 조인 및 무결성 검사 과정을 생략함으로써 대용량 처리 시 연산 리소스를 최적화했습니다.

- 에러 분류의 명확성 (Classification Clarity)  
  "단계별 거부 데이터 필터링" 요구사항을 충족하기 위해, 하나의 레코드가 서로 다른 단계의 에러를 중복 보유하지 않도록 설계했습니다.  
  이를 통해 ML 엔지니어는 데이터 파이프라인의 병목 지점(수집 장비 결함 vs 모델 추론 오류)을 명확히 식별할 수 있습니다.

---

# 🛠 3. 기술 스택 (Tech Stack) 및 선택 이유

### 3.1 Framework: FastAPI

선택 이유: 자율주행 데이터의 대용량 처리를 위해 고성능 비동기 처리가 가능한 FastAPI를 선택했습니다.  
Pydantic을 이용한 데이터 검증과 Swagger UI(/docs) 자동 생성 기능은 API 테스트 및 문서화 시간을 획기적으로 단축해 줍니다.

### 3.2 Database: SQLite3

선택 이유: 별도의 서버 설치 없이 ml_data.db 파일 하나로 관계형 데이터 정합성을 유지할 수 있고, 모든 데이터를 관리할 수 있어 이식성이 매우 뛰어납니다.  
프로토타입 개발 및 로컬 데이터 통합 환경에 가장 최적화된 선택입니다.

### 3.3 Data Analysis: Pandas

선택 이유: 복잡한 데이터 프레임의 병합(Merge), 정규화(Normalization), 그리고 거부된 데이터(Rejections)의 통계 처리를 위해 파이썬 표준 데이터 분석 라이브러리인 Pandas를 사용했습니다.  
수천 건의 JSON/CSV 데이터를 고속으로 로드하고, 벡터화 연산(Vectorized Operations)을 통해 데이터 프레임 병합 및 통계 처리를 수행합니다.

---

# 🔄 4. 데이터 분석 & DB 적재

# `POST /analyze` : 세 파일의 데이터를 분석하여 DB에 적재합니다.

### 4.1 Status

데이터 처리의 성공 여부를 나타내는 필드입니다. "success" 또는 "error"로 반환됩니다.

### 4.2 처리 요약 (Processing Summary)

전체 입력 데이터 중 결함을 걸러내고 최종 학습에 투입 가능한 유효 데이터의 총량과 정제 효율을 정량적으로 증명하는 지표입니

- Total Input Videos: 전체 영상 수
- Integrated Videos: 최종 학습 가능 데이터
- Integration Rate: 정제 성공률
- Total Rejections: 제거된 데이터 수

### 4.3 단계별 거부 (Rejection by Stage)

각 처리 단계(ODD 매칭, 라벨링 검증)별로 거절된 영상 수를 집계하여 어느 단계에서 문제가 발생하는지 파악합니다.

#### Stage 설명

- odd_tagging_step ODD 매칭 실패
- auto_labeling_step 라벨 검증 실패

### 4.4 사유별 거부 (Rejection by Reason)

거절 사유별로 집계하여 어떤 유형의 오류가 가장 빈번한지 분석합니다.

#### Stage 1 (ODD)

- missing_odd_metadata
- duplicate_odd_metadata

#### Stage 2 (Labeling)

- missing_label_data
- zero_obj_count
- negative_obj_count
- non_integer_obj_count
- duplicate_label_class

### 4.5 통계 분석 (Statistical Report)

최종 통합된 데이터셋에 대한 통계 분석을 통해 학습 데이터의 특성과 편향성을 파악합니다

- Object Class Frequency: 각 객체 클래스(예: 자동차, 보행자 등)가 전체 영상에서 얼마나 자주 등장하는지 분석하여 클래스 불균형 문제를 탐지합니다.
- Label Class Distribution: 각 객체 클래스가 전체 영상 중 몇 퍼센트의 영상에 출현하는지 분석합니다. 특정 배경에만 객체가 편중되어 학습되는 '배경 편향성'을 탐지하는 데 사용됩니다.
- Scene Complexity Distribution: 영상 내 총 객체 수를 기준으로 저/중/고밀도 상황을 분류합니다. 모델이 혼잡한 환경에서 성능이 얼마나 유지되는지 테스트하기 위한 벤치마크 데이터셋 구성의 근거가 됩니다.
- Environment Report: 기상, 시간대, 노면 상태별 비중(%)을 계산하여 학습 데이터의 편향성을 수치화합니다.
  - weather_distribution: 맑음, 비, 눈 등 다양한 기상 조건이 학습 데이터에 어떻게 분포되어 있는지 분석합니다.
  - time_of_day_distribution: 낮, 밤 등 시간대별로 학습 데이터가 어떻게 분포되어 있는지 분석합니다.
  - scenario_distribution: 기상과 시간대의 조합별로 학습 데이터가 어떻게 분포되어 있는지 분석합니다.
- Label Density Analysis (avg_labels_per_video): 영상당 평균 객체 수를 산출하여 데이터의 복잡도(Complexity)를 파악합니다.

# 🔍 5. 거부 데이터 추적 및 조회 (Rejection Tracking)

# `GET /rejections`: 정제 과정에서 제외된 데이터 목록을 상세 사유와 함께 조회합니다.

### 5.1 개요 (Overview)

POST /analyze 파이프라인 실행 중 **순차적 검증(Sequential Validation)**을 통과하지 못하고 격리된 데이터를 추적합니다. ML 엔지니어는 이 엔드포인트를 통해 원본 데이터의 결함 양상을 파악하고 수집 장비나 라벨링 모델의 이상 여부를 진단할 수 있습니다.

### 5.2 필터링 파라미터 (Query Parameters)

데이터의 양이 많을 경우를 대비하여 특정 단계나 사유별로 정밀하게 대상을 좁힐 수 있는 필터링 기능을 제공합니다.

- stage: 문제가 발생한 처리 단계를 기준으로 필터링합니다. (odd_tagging_step, auto_labeling_step)
- reason: 구체적인 거부 사유를 기준으로 필터링합니다. (예: missing_odd_metadata, zero_obj_count 등)
- page & size: 대용량 로그 조회를 위한 페이지네이션 기능을 지원합니다.

### 5.3 단계별 조회 (Filtering by Stage)

검증 파이프라인의 계층적 구조에 따라 발생 지점을 구분하여 조회합니다.

- odd_tagging_step: 영상과 주행 환경 정보(ODD) 간의 매칭이 실패한 케이스입니다.  
  데이터 수집 단계의 센서 로그 누락이나 식별자 오류를 추적할 때 사용합니다.
- auto_labeling_step: 환경 정보는 정상이나 라벨링 데이터 내부의 수치적 결함이 발견된 케이스입니다.  
  모델의 추론 정확도나 포맷 변환 로직의 오류를 진단할 때 사용합니다.

### 5.4 사유별 조회 (Filtering by Reason)

특정 결함 유형을 집중적으로 분석하기 위해 세부 사유별 조회를 지원합니다.

#### Stage 1 (ODD) 관련 사유

- missing_odd_metadata: 환경 정보 부재로 인해 학습 데이터로서의 컨텍스트를 상실한 데이터.
- duplicate_odd_metadata: 중복된 메타데이터로 인해 데이터 신뢰성이 깨진 데이터.

#### Stage 2 (Labeling) 관련 사유

- missing_label_data: 라벨 파일 자체가 존재하지 않아 학습에 활용 불가능한 데이터.
- zero_obj_count: 단 하나의 클래스라도 객체 수가 0으로 표기되어 품질이 의심되는 데이터.
- negative_obj_count / non_integer_obj_count: 논리적 혹은 형식적 수치 오류가 포함된 데이터.
- duplicate_label_class: 데이터 정합성 오류로 인해 클래스가 중복 정의된 데이터.

---

### 5.5 응답 구조 (Response Schema)

각 거부 데이터는 추적성을 위해 다음 정보를 포함하여 반환됩니다.

- video_id: 대상 영상의 고유 식별자.
- stage: 탈락이 확정된 검증 단계.
- reason: 구체적인 탈락 사유 (복합 결함 시 &로 연결된 상세 리포트 제공).
- raw_data: 문제 진단을 위해 보존된 원본 레코드 데이터 (JSON 형태).

# 🔍 6. 통합 데이터 필터 검색 (Filtered Search)

# `POST /search` : 복합 조건을 조합하여 integrated_data를 정밀 필터링합니다.

### 6.1 개요 (Overview)

단순 조회를 넘어, 자율주행 모델 학습에 필요한 **특정 시나리오(예: 비 오는 밤, 보행자 10명 이상)**를 추출하기 위한 핵심 엔드포인트입니다. 모든 필터 조건은 AND로 결합됩니다.

### 6.2 필터링 메커니즘 (Filtering Mechanism)

동등 비교 (Equivalence Matching)

- 대상: weather, time_of_day, road_surface, headlights_on, wiper_on
- 특징: 카테고리성 문자열이나 상태값(0/1)이 정확히 일치하는 데이터를 찾습니다.

수치 범위 검색 (Numeric Range & Sandwich)

- 대상: video_id, temperature_celsius, wiper_level 및 객체 카운트 등
- 접미사: {컬럼명}\_min (이상), {컬럼명}\_max (이하)
- 온도나 신뢰도 같은 실수형(Float) 데이터는 = 비교 대신 \_min과 \_max를 동시에 사용하는 범위를 지정하는 것이 정확합니다.

언패킹된 라벨 필터 (Unpacked Label Search)

- 객체 수: label\_{class}\_min/max (예: label_car_min: 30)
- 신뢰도: label\_{class}\_confidence_min/max (예: label_pedestrian_confidence_min: 0.8)

### 6.3 🕒 시간 데이터 매칭 특화 (Time-Series Substr Match)

- recorded_at 및 labeled_at 검색 시, 시스템은 사용자가 입력한 문자열의 길이를 감지하여 DB 값의 앞부분과 비교합니다.
- 2026-01: 2026년 1월 데이터 전체 검색
- 2026-01-10T19: 2026년 1월 10일 19시 데이터 검색

### 6.4 요청 예시 (Request Sample)

"Video #3"를 찾는 request 샘플입니다.

```JSON
{
"weather": "sunny",
"time_of_day": "night",
"road_surface": "dry",
"wiper_on": 1,
"headlights_on": 1,
"temperature_celsius_min": 14.5,
"temperature_celsius_max": 14.6,
"label_car_min": 31,
"label_car_confidence_min": 0.83,
"label_pedestrian_min": 26,
"recorded_at_min": "2026-01-10T00:00:00",
"recorded_at_max": "2026-01-12T23:59:59"
}
```

### 6.5 응답 구조 (Response Sample)

페이지네이션 정보와 함께 필터링된 결과 리스트를 반환합니다.

```JSON
{
"status": "success",
"pagination": {
  "page": 1,
  "size": 50,
  "total_found": 1
}
  "results": [
    {
    "video_id": 3,
    "recorded_at": "2026-01-10T19:44:32+0900",
    "temperature_celsius": 14.5555,
    "weather": "sunny",
    "labels": {
    "car": { "count": 31, "avg_confidence": 0.831 },
    "pedestrian": { "count": 26, "avg_confidence": 0.738 }
    },
    "label_car_count": 31,
    "label_car_confidence": 0.831
    }
  ]
}
```

# 📡 7. API 엔드포인트 상세 실행 가이드

본 시스템은 FastAPI의 자동 문서화 기능을 통해 별도의 클라이언트 없이도 웹 브라우저(Swagger UI)에서 즉시 테스트가 가능합니다.

## 7.1 환경 구축 및 서버 실행 (Quick Start)

데이터 분석 및 API 서빙을 위한 필수 라이브러리를 설치하고 서버를 가동합니다.

## 1. 의존성 설치

pip install fastapi uvicorn pandas

## 2. 서버 실행 (프로젝트 루트 디렉토리 기준)

uvicorn app.main:app --reload

## 3. API 문서 접속

## 서버 실행 후 브라우저에서 아래 주소로 접속합니다. http://127.0.0.1:8000/docs

7.2 주요 엔드포인트 활용법
1️⃣ 데이터 파이프라인 구동: POST /analyze
모든 데이터 프로세싱의 시작점입니다. 분산된 CSV/JSON 소스를 읽어 무결성을 검증하고 관계형 DB(SQLite)에 적재합니다.

실행 방법: Swagger UI에서 Try it out ➔ Execute 버튼 클릭.

작동 원리:

Stage 1 (ODD): 환경 메타데이터 매칭 및 중복 검사.

Stage 2 (Labeling): 객체 카운트 논리 오류(음수, 실수, 누락 등) 검사.

결과: 통합 성공 수치, 정제 효율(%), 사유별 거절 통계 리포트 반환.

2️⃣ 오류 데이터 정밀 추적: GET /rejections
검증 파이프라인에서 탈락한 데이터를 상세 사유와 함께 조회하여 데이터 품질을 진단합니다.

주요 파라미터:

stage: 특정 검증 단계(odd_tagging_step, auto_labeling_step)별 필터링.

reason: 구체적 오류 코드(zero_obj_count, missing_odd_metadata 등) 검색.

활용 팁: raw_data 필드를 확인하여 원본 데이터의 어떤 부분이 규격에 맞지 않는지 즉각적인 디버깅이 가능합니다.

3️⃣ 고성능 조건부 검색: POST /search
정제 완료된 integrated_data를 대상으로 ML 모델 학습에 필요한 최적의 데이터셋을 추출합니다.

요청 예시 (Scenario: High Density Sunny Night):

JSON
{
"weather": "sunny",
"time_of_day": "night",
"video_id_min": 1, "video_id_max": 100,
"label_car_min": 30,
"label_pedestrian_min": 10,
"recorded_at_min": "2026-03-01"
}
핵심 기능: 모든 수치형 필드에 대해 \_min, \_max 접미사를 통한 범위(Sandwich) 검색을 지원하며, 라벨 데이터 내의 특정 클래스 개수별 필터링이 가능합니다.

7.3 운영 팁 (Best Practices)
초기화: 데이터 소스(selections.json 등)가 변경된 경우, 항상 POST /analyze를 재실행하여 DB를 최신화하십시오.

페이지네이션: 대량의 데이터 조회 시 page와 size 파라미터를 활용하여 네트워크 부하를 최소화하십시오.

데이터 백업: 생성된 ml_data.db 파일은 로컬 SQLite 툴(예: DBeaver, DB Browser for SQLite)을 통해 직접 쿼리하거나 백업할 수 있습니다.

# 📡 7. API 엔드포인트 상세 실행 가이드

본 시스템은 FastAPI의 자동 문서화 기능을 통해 별도의 클라이언트 없이도 웹 브라우저(Swagger UI)에서 즉시 테스트가 가능합니다.

---

## 7.1 환경 구축 및 서버 실행 (Quick Start)

데이터 분석 및 API 서빙을 위한 필수 라이브러리를 설치하고 서버를 가동합니다.

### 1. 의존성 설치

```bash
pip install fastapi uvicorn pandas
2. 서버 실행 (프로젝트 루트 디렉토리 기준)
uvicorn app.main:app --reload
3. API 문서 접속

서버 실행 후 브라우저에서 아래 주소로 접속합니다.
http://127.0.0.1:8000/docs

7.2 주요 엔드포인트 활용법
1️⃣ 데이터 파이프라인 구동: POST /analyze

모든 데이터 프로세싱의 시작점입니다. 분산된 CSV/JSON 소스를 읽어 무결성을 검증하고 관계형 DB(SQLite)에 적재합니다.

실행 방법: Swagger UI에서 Try it out ➔ Execute 버튼 클릭.
작동 원리:
Stage 1 (ODD): 환경 메타데이터 매칭 및 중복 검사.
Stage 2 (Labeling): 객체 카운트 논리 오류(음수, 실수, 누락 등) 검사.
결과: 통합 성공 수치, 정제 효율(%), 사유별 거절 통계 리포트 반환.
2️⃣ 오류 데이터 정밀 추적: GET /rejections

검증 파이프라인에서 탈락한 데이터를 상세 사유와 함께 조회하여 데이터 품질을 진단합니다.

주요 파라미터:
stage: 특정 검증 단계(odd_tagging_step, auto_labeling_step)별 필터링.
reason: 구체적 오류 코드(zero_obj_count, missing_odd_metadata 등) 검색.
활용 팁:
raw_data 필드를 확인하여 원본 데이터의 어떤 부분이 규격에 맞지 않는지 즉각적인 디버깅이 가능합니다.
3️⃣ 고성능 조건부 검색: POST /search

정제 완료된 integrated_data를 대상으로 ML 모델 학습에 필요한 최적의 데이터셋을 추출합니다.

요청 예시 (Scenario: High Density Sunny Night)
{
  "weather": "sunny",
  "time_of_day": "night",
  "video_id_min": 1,
  "video_id_max": 100,
  "label_car_min": 30,
  "label_pedestrian_min": 10,
  "recorded_at_min": "2026-03-01"
}
핵심 기능:
모든 수치형 필드에 대해 _min, _max 접미사를 통한 범위(Sandwich) 검색을 지원하며,
라벨 데이터 내의 특정 클래스 개수별 필터링이 가능합니다.
7.3 운영 팁 (Best Practices)
초기화:
데이터 소스(selections.json 등)가 변경된 경우, 항상 POST /analyze를 재실행하여 DB를 최신화하십시오.
페이지네이션:
대량의 데이터 조회 시 page와 size 파라미터를 활용하여 네트워크 부하를 최소화하십시오.
데이터 백업:
생성된 ml_data.db 파일은 로컬 SQLite 툴(예: DBeaver, DB Browser for SQLite)을 통해 직접 쿼리하거나 백업할 수 있습니다.
```
