from collect import *
from merge import *
from preprocess import *
from model import *
import config

n_iter_count = config.n_iter_count
landscape = config.landscape 

def run_pipeline():
    
   # 데이터 수집 단계
    
   # 사용자 입력을 통해 데이터 수집 여부 결정
    fetch_data = config.fetch_data
    
    if fetch_data == 'y':
        # 데이터 수집 단계
        before_path = config.before_path  ## before_path에 수집이 이미 완료된 파일(daily_data.xlsx)를 지정해주세요
        df_2 = pd.read_excel(before_path, engine="openpyxl")
        weather_data = fetch_and_process_all_years(df_2)    
    else:
        weather_data = pd.read_excel(config.BASE_DIR + "/machine_learning_project/data/weather/2015_2024_mosquito.xlsx")
    
    # 데이터 병합 단계
    data = merge_data(weather_data)
    data = data.reset_index(drop=True)

    #경관요소 선택 단계
    landscape = config.landscape
    
    if landscape == 1 :
        data = data[data['landscape'] == 1]
    elif landscape == 2 :
        data = data[data['landscape'] == 2]
    elif landscape == 3 :
        data = data[data['landscape'] == 3]
    else :
        data = data
    
    # 데이터 전처리 단계
    scaler, X_train_scaled, X_test_scaled, y_train, y_test, X_train, train_data_sorted, test_data_sorted = processed_data(data)
    
    print("전처리_완료")
    print("모델 학습 시작")
    
    save_path = config.save_path
    n_iter_count = config.n_iter_count
    region = config.region
    
    results = run_models(n_iter_count, save_path, region, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted, landscape)
    return results

if __name__ == "__main__" :
    run_pipeline()