import os

# 기본 디렉토리 설정
BASE_DIR = "F:/박정현/ML"

# 데이터 수집 관련 설정
'''
- fetch_data: 데이터 수집 여부를 결정하는 변수
    - 'y': 새로운 데이터를 수집하여 기존 데이터를 대체
    - 'n': 기존 데이터를 사용하고 새로운 데이터는 수집하지 않음
- before_path: 기존 기상 데이터 파일의 경로
- DMS_CODE_PATH: DMS 관측소 정보를 포함하는 매핑 파일 경로
'''
fetch_data = 'n'
before_path = r'F:/박정현/ML/machine_learning_project/data/weather/2015_2023_daily_data.xlsx'
DMS_CODE_PATH = r'F:/박정현/ML/machine_learning_project/data/DMS_관측소_매핑.xlsx'

# API 기본 설정
'''
- API_DOMAIN: 기상 데이터 요청을 위한 기본 API URL
- API_VAR: API 요청 시 사용되는 변수
- API_AUTH_KEY: API 접근을 위한 인증 키 (보안 유지 필요)
자세한 항목은 'https://apihub.kma.go.kr/' 참고
'''
API_DOMAIN = "https://apihub.kma.go.kr/api/typ01/url/awsh.php?"
API_VAR = 'var='
API_AUTH_KEY = "s5DFRYyOQqKQxUWMjlKi9g"

# 데이터 수집 기간 설정
'''
- start_year: 데이터 수집을 시작할 연도
- end_year: 데이터 수집을 종료할 연도
- TEST_YEAR: 모델 검증을 위한 테스트 데이터 연도 (해당 연도 이후 데이터를 테스트셋으로 사용)
'''
start_year = 2015
end_year = 2024
TEST_YEAR = 2024

# 모기 데이터 관련 설정
'''
- MOSQUITO_PATH: 모기 개체 수 데이터를 저장하는 경로
'''
MOSQUITO_PATH = r"F:/박정현/ML/machine_learning_project/data/mosquito"

# 환경 설정
'''
- landscape: 경관 요소 필터링 옵션 (1: 특정 환경, 0: 전체 환경)
- save_path: 모델 학습 결과 및 시각화 파일 저장 경로
- n_iter_count: 하이퍼파라미터 최적화 반복 횟수
'''
landscape = 2
save_path = r'F:/박정현/ML/machine_learning_project/result_2024'
n_iter_count = 2
# region = "1" 미사용 변수 삭제

# 실행할 모델 설정
'''
- MODELS_TO_RUN: 실행할 머신러닝 모델을 리스트로 지정
  ['lgbm', 'rf', 'gb', 'xgb'] 중 원하는 모델을 선택하여 실행
   "lgbm": "Light Gradient Boosting",
    "xgb": "Extreme Gradient Boosting",
    "rf": "Random Forest",
    "gb": "Gradient Boosting"
'''
MODELS_TO_RUN = ['lgbm']

# 📌 Google Drive 업로드 설정
'''
- GOOGLE_CREDENTIALS_FILE: 구글 드라이브 API 인증을 위한 크리덴셜 파일 경로
- GOOGLE_TOKEN_FILE: 구글 드라이브 API 인증 토큰 파일 경로
- GOOGLE_FOLDER_ID: Google Drive에서 파일을 업로드할 폴더의 ID
'''
GOOGLE_CREDENTIALS_FILE = "F:/박정현/ML/machine_learning_project/scripts/credentials.json"
GOOGLE_TOKEN_FILE = "F:/박정현/ML/machine_learning_project/scripts/token.json"
GOOGLE_FOLDER_ID = "1GbNbQta8EGl7G853K9Cey6dd5NDzVDf6"

# 하이퍼파라미터 설정
'''
각 머신러닝 모델별로 최적화할 하이퍼파라미터 범위를 설정한다.
- n_estimators: 생성할 트리 개수
- max_depth: 트리의 최대 깊이
- learning_rate: 학습 속도를 조절하는 파라미터
- num_leaves: LightGBM 모델에서 사용되는 리프 노드 개수
- min_child_samples: 리프 노드에서 요구되는 최소 샘플 수
- subsample: 일부 데이터를 사용하여 모델이 과적합되지 않도록 하는 비율
- colsample_bytree: 트리 학습 시 사용할 특성의 비율
- min_samples_split: 내부 노드를 분할하기 위한 최소 샘플 개수
- min_samples_leaf: 리프 노드(끝 노드)에 필요한 최소 샘플 개수
'''
HYPERPARAMS = {
    "lgbm": {  # LightGBM 모델
        'n_estimators': (400, 700, 20),
        'max_depth': (5, 40, 5),
        'learning_rate': (0.005, 0.05),
        'num_leaves': (15, 50, 5),
        'min_child_samples': (10, 30, 2)
    },
    "xgb": {  # XGBoost 모델
        'n_estimators': (300, 700, 10),
        'max_depth': (3, 15, 2),
        'learning_rate': (0.005, 0.04),
        'subsample': (0.5, 0.85),
        'colsample_bytree': (0.4, 0.8)
    },
    "rf": {  # Random Forest 모델
        'n_estimators': (300, 500, 10),
        'max_depth': (30, 50, 5),
        'min_samples_split': (5, 10, 1),
        'min_samples_leaf': (5, 10, 1)
    },
    "gb": {  # Gradient Boosting 모델
        'n_estimators': (300, 500, 10),
        'max_depth': (4, 20, 2),
        'min_samples_split': (3, 15, 2),
        'min_samples_leaf': (1, 10, 1),
        'learning_rate': (0.01, 0.05),
        'subsample': (0.7, 0.9)
    }
}



