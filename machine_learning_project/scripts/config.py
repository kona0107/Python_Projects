# 기본 디렉토리(루트 디렉토리를 입력해주세요)
BASE_DIR = "F:/박정현/ML"

# 데이터 수집 결정(y, n)
fetch_data = 'n'

# 전년도 데이터
before_path = r'F:/박정현/ML/machine_learning_project/data/weather/2015_2023_daily_data.xlsx'
DMS_CODE_PATH = r'F:/박정현/ML/machine_learning_project/data/DMS_관측소_매핑.xlsx'

# API 기본 URL
domain = "https://apihub.kma.go.kr/api/typ01/url/awsh.php?"
var = 'var='
option = "authKey=s5DFRYyOQqKQxUWMjlKi9g"  # 인증키

# 데이터 수집 시작 년도, 종료 년도
start_year = 2024
end_year = 2024

#테스트 데이터 년도
TEST_YEAR = 2024

#모기 데이터 합치기
MOSQUITO_PATH = r"F:/박정현/ML/machine_learning_project/data/mosquito"

# 경관요소 선택(0(전체), 1, 2, 3)
landscape = 1

save_path = r'F:/박정현/ML/machine_learning_project/models'

n_iter_count = 100

region = "1"

# 실행할 모델을 선택하는 리스트 ['lgbm', 'rf', 'gb', 'xgb']
MODELS_TO_RUN = ['lgbm', 'rf', 'gb', 'xgb']

# 하이퍼파라미터 조정

HYPERPARAMS = {
    "lgbm": {
        'n_estimators': (400, 600, 10),
        'max_depth': (10, 60, 5),
        'learning_rate': (0.01, 0.05),
        'num_leaves': (20, 80, 5),
        'min_child_samples': (3, 25, 2)
    },
    "rf": {
        'n_estimators': (300, 700, 10),
        'max_depth': (10, 60, 5),
        'min_samples_split': (3, 15, 2),
        'min_samples_leaf': (1, 10, 1)
    },
    "gb": {
        'n_estimators': (400, 600, 10),
        'max_depth': (4, 20, 2),
        'min_samples_split': (3, 15, 2),
        'min_samples_leaf': (1, 10, 1),
        'learning_rate': (0.01, 0.05)
    },
    "xgb": {
        'n_estimators': (300, 600, 10),
        'max_depth': (4, 20, 2),
        'learning_rate': (0.01, 0.05),
        'subsample': (0.6, 0.9),
        'colsample_bytree': (0.5, 0.9)
    }
}

