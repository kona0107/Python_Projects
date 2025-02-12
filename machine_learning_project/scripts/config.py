import os

# 기본 디렉토리 설정
BASE_DIR = "F:/박정현/ML"

# 데이터 수집 관련 설정
fetch_data = 'n'
before_path = r'F:/박정현/ML/machine_learning_project/data/weather/2015_2023_daily_data.xlsx'
DMS_CODE_PATH = r'F:/박정현/ML/machine_learning_project/data/DMS_관측소_매핑.xlsx'

# API 기본 설정
API_DOMAIN = "https://apihub.kma.go.kr/api/typ01/url/awsh.php?"
API_VAR = 'var='
API_AUTH_KEY = "s5DFRYyOQqKQxUWMjlKi9g"

#  데이터 수집 기간
start_year = 2024
end_year = 2024
TEST_YEAR = 2024

# 모기 데이터 경로
MOSQUITO_PATH = r"F:/박정현/ML/machine_learning_project/data/mosquito"

# 환경 설정
landscape = 1
save_path = r'F:/박정현/ML/machine_learning_project/final_2024' # models-> final 2024로 변경함
n_iter_count = 100
region = "1"

# 실행할 모델 설정
MODELS_TO_RUN = ['gb']

# 📌 Google Drive 업로드 설정 
GOOGLE_CREDENTIALS_FILE = "F:/박정현/ML/machine_learning_project/scripts/credentials.json"
GOOGLE_TOKEN_FILE = "F:/박정현/ML/machine_learning_project/scripts/token.json"
GOOGLE_FOLDER_ID = "1GbNbQta8EGl7G853K9Cey6dd5NDzVDf6"

# 하이퍼파라미터 설정
HYPERPARAMS = {
    "lgbm": {
        'n_estimators': (400, 700, 20),
        'max_depth': (5, 40, 5),
        'learning_rate': (0.005, 0.05),
        'num_leaves': (15, 50, 5),
        'min_child_samples': (10, 30, 2)
    },
    "xgb": {
        'n_estimators': (300, 700, 10),
        'max_depth': (3, 15, 2),
        'learning_rate': (0.005, 0.04),
        'subsample': (0.5, 0.85),
        'colsample_bytree': (0.4, 0.8)
    },
    "rf": {
        'n_estimators': (300, 500, 10),
        'max_depth': (30, 50, 5),
        'min_samples_split': (5, 10, 1),
        'min_samples_leaf': (5, 10, 1)
    },
    "gb": {
        'n_estimators': (300, 500, 10),
        'max_depth': (4, 20, 2),
        'min_samples_split': (3, 15, 2),
        'min_samples_leaf': (1, 10, 1),
        'learning_rate': (0.01, 0.05),
        'subsample': (0.7, 0.9)
    }
}



