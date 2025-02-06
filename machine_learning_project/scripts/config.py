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
MODELS_TO_RUN = ['rf', 'gb']

# 하이퍼파라미터 조정
HYPERPARAMS = {
    "lgbm": {
        'n_estimators': (400, 700, 20),  # 탐색 범위 확장
        'max_depth': (5, 40, 5),  # 깊이 제한으로 과적합 방지
        'learning_rate': (0.005, 0.05),  # 세밀한 튜닝
        'num_leaves': (15, 50, 5),  # 너무 깊은 트리 방지
        'min_child_samples': (10, 30, 2)  # 최소 샘플 개수 증가
    },
    "xgb": {
        'n_estimators': (300, 700, 10),  # 더 많은 탐색
        'max_depth': (3, 15, 2),  # 깊이 제한으로 과적합 방지
        'learning_rate': (0.005, 0.04),  # 세밀한 튜닝
        'subsample': (0.5, 0.85),  # 과적합 방지
        'colsample_bytree': (0.4, 0.8)  # feature 사용 개수 제한
    },
    "rf": {
        'n_estimators': (300, 500, 10),  # 트리 개수를 줄여 속도 향상
        'max_depth': (10, 60, 5),  # 너무 깊은 트리 방지
        'min_samples_split': (3, 15, 2),  # 최소 분할 샘플 증가
        'min_samples_leaf': (1, 10, 1),  # 리프 노드 최소 샘플 증가
    },
    "gb": {
        'n_estimators': (300, 500, 10),  # 트리 개수를 500개 이하로 줄이면서 최적 성능 확보
        'max_depth': (4, 20, 2),  # 깊이를 제한하여 과적합 방지
        'min_samples_split': (3, 15, 2),  # 너무 작은 샘플로 분할하지 않도록 설정
        'min_samples_leaf': (1, 10, 1),  # 리프 노드 최소 샘플 개수 제한
        'learning_rate': (0.01, 0.05),  # 안정적인 학습률
        'subsample': (0.7, 0.9)  # 일부 데이터만 사용하여 속도 향상
    }
}
