🚗 자율주행 데이터 통합 및 검색 API 시스템 (ML Data SQL API)
본 프로젝트는 자율주행 차량에서 수집된 센서 데이터(odds.csv), 객체 인식 결과(labels.csv), 그리고 원본 선택 (selections.json) 데이터를 통합하여 정제된 학습용 데이터셋을 구축하고 관리하는 RESTful API 서버입니다.

📋 1. 프로젝트 개요
자율주행 데이터는 날씨, 온도 등 정형 데이터와 객체 카운트와 같은 비정형 데이터가 혼재되어 있습니다. 본 시스템은 데이터 파이프라인(analyze)을 통해 오류 데이터를 자동으로 분류(Rejection)하고, 통합된 데이터를 사용자가 원하는 정밀한 조건(Sandwich Search)으로 검색할 수 있게 설계되었습니다.

🧠 2. 설계 철학 및 데이터 파이프라인
2.1 순차적 검증 파이프라인 (Sequential Validation Pipeline)

데이터 신뢰성을 보장하기 위해 단계별 검증 구조를 채택:

핵심 원칙
ODD 우선 원칙
환경 정보(날씨, 시간 등)가 없는 데이터는 무조건 제외
연산 최적화
초기 단계에서 실패한 데이터는 이후 단계 생략
명확한 에러 분류
각 데이터는 하나의 명확한 실패 원인만 가짐

2.2 데이터 처리 흐름
Raw Data
→ ODD Tagging (Stage 1)
→ Auto Labeling Validation (Stage 2)
→ Integrated Data (정상)
→ Rejections (오류 데이터)

본 시스템은 데이터의 신뢰성을 보장하기 위해 **순차적 검증 파이프라인(Sequential Validation Pipeline)**을 채택하였습니다. 이는 상위 단계의 데이터가 무결하지 않을 경우 하위 단계의 검증을 생략하고 즉시 격리하는 아키텍처입니다.

순차적 검증 로직 (Sequential Validation Logic)
계층적 데이터 무결성 (ODD 우선 원칙): 자율주행 도메인에서 **주행 환경 정보(ODD)**는 객체 라벨링 데이터의 전제 조건입니다. 기상, 시간, 노면 상태 등 컨텍스트가 결여된 라벨 데이터는 모델 학습용으로 부적합하므로, Stage 1 (ODD Tagging) 검증을 최우선 게이트웨이로 배치했습니다.

연산 효율성 및 처리 최적화: 데이터셋 분석 결과, 2,000건 이상의 거부 사례가 ODD와 라벨 데이터의 동시 누락과 관련되어 있습니다. 순차적 접근 방식을 통해 1단계에서 결함이 발견된 레코드는 2단계(Auto Labeling) 조인 및 무결성 검사 과정을 생략함으로써 대용량 처리 시 연산 리소스를 최적화했습니다.

에러 분류의 명확성 (Classification Clarity): "단계별 거부 데이터 필터링" 요구사항을 충족하기 위해, 하나의 레코드가 서로 다른 단계의 에러를 중복 보유하지 않도록 설계했습니다. 이를 통해 ML 엔지니어는 데이터 파이프라인의 병목 지점(수집 장비 결함 vs 모델 추론 오류)을 명확히 식별할 수 있습니다.

🛠 3. 기술 스택 (Tech Stack) 및 선택 이유

3.1. Framework: FastAPI
선택 이유: 자율주행 데이터의 대용량 처리를 위해 고성능 비동기 처리가 가능한 FastAPI를 선택했습니다. Pydantic을 이용한 데이터 검증과 Swagger UI(/docs) 자동 생성 기능은 API 테스트 및 문서화 시간을 획기적으로 단축해 줍니다.

3.2. Database: SQLite3
선택 이유: 별도의 서버 설치 없이 ml_data.db 파일 하나로 관계형 데이터 정합성을 유지할 수 있고, 모든 데이터를 관리할 수 있어 이식성이 매우 뛰어납니다. 프로토타입 개발 및 로컬 데이터 통합 환경에 가장 최적화된 선택입니다.

3.3. Data Analysis: Pandas
선택 이유: 복잡한 데이터 프레임의 병합(Merge), 정규화(Normalization), 그리고 거부된 데이터(Rejections)의 통계 처리를 위해 파이썬 표준 데이터 분석 라이브러리인 Pandas를 사용했습니다. 수천 건의 JSON/CSV 데이터를 고속으로 로드하고, 벡터화 연산(Vectorized Operations)을 통해 데이터 프레임 병합 및 통계 처리를 수행합니다.

🔄 4. 데이터 분석 & DB 적재

#### `POST /analyze` : 세 파일의 데이터를 분석하여 DB에 적재합니다.

4.1 Status: 데이터 처리의 성공 여부를 나타내는 필드입니다. "success" 또는 "error"로 반환됩니다.

4.2 처리 요약 (Processing Summary): 전체 입력 데이터 중 결함을 걸러내고 최종 학습에 투입 가능한 유효 데이터의 총량과 정제 효율을 정량적으로 증명하는 지표입니
Total Input Videos: 전체 영상 수
Integrated Videos: 최종 학습 가능 데이터
Integration Rate: 정제 성공률
Total Rejections: 제거된 데이터 수

4.3 단계별 거부 (Rejection by Stage): 각 처리 단계(ODD 매칭, 라벨링 검증)별로 거절된 영상 수를 집계하여 어느 단계에서 문제가 발생하는지 파악합니다.

Stage 설명

- odd_tagging_step ODD 매칭 실패
- auto_labeling_step 라벨 검증 실패

  4.4 사유별 거부 (Rejection by Reason): 거절 사유별로 집계하여 어떤 유형의 오류가 가장 빈번한지 분석합니다.
  Stage 1 (ODD)

- missing_odd_metadata
- duplicate_odd_metadata
  Stage 2 (Labeling)
- missing_label_data
- zero_obj_count
- negative_obj_count
- non_integer_obj_count
- duplicate_label_class

  4.5 통계 분석 (Statistical Report): 최종 통합된 데이터셋에 대한 통계 분석을 통해 학습 데이터의 특성과 편향성을 파악합니다

- Object Class Frequency: 각 객체 클래스(예: 자동차, 보행자 등)가 전체 영상에서 얼마나 자주 등장하는지 분석하여 클래스 불균형 문제를 탐지합니다.
- Label Class Distribution: 각 객체 클래스가 전체 영상 중 몇 퍼센트의 영상에 출현하는지 분석합니다. 특정 배경에만 객체가 편중되어 학습되는 '배경 편향성'을 탐지하는 데 사용됩니다.
- Scene Complexity Distribution: 영상 내 총 객체 수를 기준으로 저/중/고밀도 상황을 분류합니다. 모델이 혼잡한 환경에서 성능이 얼마나 유지되는지 테스트하기 위한 벤치마크 데이터셋 구성의 근거가 됩니다.
- Environment Report: 기상, 시간대, 노면 상태별 비중(%)을 계산하여 학습 데이터의 편향성을 수치화합니다.
  - weather_distribution: 맑음, 비, 눈 등 다양한 기상 조건이 학습 데이터에 어떻게 분포되어 있는지 분석합니다.
  - time_of_day_distribution: 낮, 밤 등 시간대별로 학습 데이터가 어떻게 분포되어 있는지 분석합니다.
  - scenario_distribution: 기상과 시간대의 조합별로 학습 데이터가 어떻게 분포되어 있는지 분석합니다.
- Label Density Analysis (avg_labels_per_video): 영상당 평균 객체 수를 산출하여 데이터의 복잡도(Complexity)를 파악합니다.

👉 데이터 편향 및 학습 품질 분석 가능

🔍 핵심 검색 로직: 부등호 기준 (Comparison Logic)
모든 수치형 데이터는 **이상(>=) 및 이하(<=)**를 기준으로 작동합니다

정확한 일치 (Exact Match): \_min과 \_max 필드에 동일한 값을 입력합니다. (예: wiper_level_min: 3, wiper_level_max: 3)

이하/이상 검색 (Threshold): 한쪽 필드만 입력합니다. (예: label_car_min: 31 → 차량이 31대 이상인 데이터)

범위 검색 (Range): 두 값을 다르게 입력하여 사이 구간을 검색합니다.

📋 검색 가능한 컬럼 및 샘플 데이터 (Searchable Fields)
아래 목록을 복사하여 POST /search의 Request Body에 바로 사용할 수 있습니다.

1. 직접 일치 및 부분 일치 필드 (String & Boolean)
   문자열은 대소문자를 구분하지 않으며, source_path는 파일명의 일부만 입력해도 검색됩니다.

   JSON
   {
   "weather": "sunny",
   "time_of_day": "night",
   "road_surface": "dry",
   "headlights_on": 1,
   "wiper_on": 1,
   "source_path": "00003.mp4"
   }

2. 수치 범위 검색 필드 (Numeric Sandwich)
   영상 정보 및 환경 센서 데이터입니다. \_min 또는 \_max 접미사를 붙여 사용합니다.
   JSON
   {
   "video\*id_min": 1,
   "video_id_max": 100,
   "id_min": 1,
   "id_max": 100,
   "temperature_fahrenheit_min": 50.0,
   "temperature_fahrenheit_max": 80.0,
   "temperature_celsius_min": 10.0,
   "temperature_celsius_max": 25.0,
   "wiper_level_min": 1,
   "wiper_level_max": 3,
   "recorded_at_min": "2026-01-01",
   "recorded_at_max": "2026-12-31"
   }
3. 객체 인식 라벨 카운트 필드 (Label Counts)
   인식된 객체의 종류별 개수입니다. label\*{객체명}\_min/max 형식을 사용합니다.
   JSON
   {
   "label_car_min": 10,
   "label_car_max": 50,
   "label_pedestrian_min": 5,
   "label_traffic_sign_min": 1,
   "label_truck_min": 0,
   "label_bus_min": 0,
   "label_motorcycle_min": 0,
   "label_cyclist_min": 0,
   "label_traffic_light_min": 0
   }
   📡 API 엔드포인트 상세 실행 가이드
   서버 실행 후 Swagger UI (http://127.0.0.1:8000/docs) 접속을 권장합니다.

1️⃣ 데이터 파이프라인 실행: POST /analyze
모든 작업의 시작점입니다. 원본 데이터를 로드하고 통합 DB를 생성합니다.

실행 방법: Execute 버튼 클릭.

작동 원리: odd_tagging 단계와 auto_labeling 단계를 거쳐 메타데이터가 누락되거나 중복된 데이터를 rejections 테이블로 분리하고, 정상 데이터는 integrated_data에 저장합니다.

결과: 통합된 데이터 개수와 거부 사유별 통계(Breakdown)를 반환합니다.

2️⃣ 통합 데이터 샘플 조회: GET /joined_data
파이프라인을 통과한 최종 통합 데이터를 확인합니다.

실행 방법: URL 직접 접속 또는 Swagger UI 실행.

작동 원리: 통합된 테이블에서 상위 50개의 레코드를 가져오며, JSON 형태의 레이블 데이터가 올바르게 파싱되었는지 확인할 수 있습니다.

3️⃣ 거부 데이터 추적: GET /rejections
정제 과정에서 제외된 데이터의 사유를 조회합니다.

파라미터: reason(거부 사유), stage(발생 단계), page, size.

실행 예시: reason=missing_odd_metadata 입력 시 메타데이터가 없어 제외된 리스트만 출력됩니다. (페이지네이션 지원)

4️⃣ 조건부 정밀 검색: POST /search
가장 강력한 기능으로, 복합적인 환경 조건을 필터링합니다.

입력 예시 (Video ID 3 찾기):

JSON
{
"video_id_min": 3, "video_id_max": 3,
"weather": "sunny",
"wiper_level_min": 3, "wiper_level_max": 3,
"label_car_min": 31, "label_car_max": 31,
"label_pedestrian_min": 11
}
특징: 모든 수치형 데이터와 라벨 카운트 필드에 대해 \_min, \_max 접미사를 사용하여 유연한 쿼리가 가능합니다.

⚙️ 설치 및 실행 (Quick Start)
의존성 설치:

Bash
pip install fastapi uvicorn pandas
서버 실행:

Bash
uvicorn app.main:app --reload
API 문서 접속:
http://127.0.0.1:8000/docs
