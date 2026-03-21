🚗 자율주행 데이터 통합 및 검색 API 시스템 (ML Data SQL API)
본 프로젝트는 자율주행 차량에서 수집된 센서 데이터(ODD), 객체 인식 결과(Labels), 그리고 원본 선택 데이터를 통합하여 정제된 학습용 데이터셋을 구축하고 관리하는 RESTful API 서버입니다.

📋 프로젝트 개요
자율주행 데이터는 날씨, 온도 등 정형 데이터와 객체 카운트와 같은 비정형 데이터가 혼재되어 있습니다. 본 시스템은 데이터 파이프라인(analyze)을 통해 오류 데이터를 자동으로 분류(Rejection)하고, 통합된 데이터를 사용자가 원하는 정밀한 조건(Sandwich Search)으로 검색할 수 있게 설계되었습니다.

🛠 기술 스택 (Tech Stack) 및 선택 이유

1. Framework: FastAPI
   선택 이유: 자율주행 데이터의 대용량 처리를 위해 고성능 비동기 처리가 가능한 FastAPI를 선택했습니다. Pydantic을 이용한 데이터 검증과 Swagger UI(/docs) 자동 생성 기능은 API 테스트 및 문서화 시간을 획기적으로 단축해 줍니다.

2. Database: SQLite3
   선택 이유: 별도의 서버 설치 없이 ml_data.db 파일 하나로 모든 데이터를 관리할 수 있어 이식성이 매우 뛰어납니다. 프로토타입 개발 및 로컬 데이터 통합 환경에 가장 최적화된 선택입니다.

3. Data Analysis: Pandas
   선택 이유: 복잡한 데이터 프레임의 병합(Merge), 정규화(Normalization), 그리고 거부된 데이터(Rejections)의 통계 처리를 위해 파이썬 표준 데이터 분석 라이브러리인 Pandas를 사용했습니다.

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
