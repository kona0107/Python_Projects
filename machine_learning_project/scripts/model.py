import numpy as np
import pandas as pd
import os
import joblib
import sys
import random
import multiprocessing
import re
import config
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import fmin, tpe, Trials, STATUS_OK
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import hyperopt.hp as hp 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def get_param_space(model_name):
    """ 
    config.py에서 하이퍼파라미터를 가져와 hyperopt 형태로 변환하는 함수
    
    매개변수:
        model_name (str): 사용할 모델 이름 (예: 'lgbm', 'rf', 'xgb', 'gb')
    
    반환값:
        dict: hyperopt 라이브러리에서 사용할 하이퍼파라미터 공간
    """
    param_ranges = config.HYPERPARAMS[model_name] # 모델별 하이퍼파라미터 범위 가져오기
    
    param_space = {}
    for key, value in param_ranges.items():
        if len(value) == 3:  # quniform (정수형)
            param_space[key] = hp.quniform(key, *value)
        elif len(value) == 2:  # loguniform (학습률 등)
            param_space[key] = hp.uniform(key, value[0], value[1])  # loguniform 대신 uniform 사용
    
    return param_space


def train_model(model_cls, param_space, n_iter_count, save_path, model_name, 
                X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, 
                test_data_sorted, landscape):
    """ 
    주어진 하이퍼파라미터 공간을 기반으로 모델을 학습하는 함수
    
    매개변수:
        model_cls (class): 사용할 모델 클래스 (예: LGBMRegressor, RandomForestRegressor 등)
        param_space (dict): hyperopt에서 사용할 하이퍼파라미터 공간
        n_iter_count (int): 하이퍼파라미터 최적화 반복 횟수
        save_path (str): 학습된 모델 저장 경로
        model_name (str): 모델 이름 (예: 'lgbm', 'rf', 'gb', 'xgb')
        X_train_scaled (pd.DataFrame): 정규화된 훈련 데이터
        X_test_scaled (pd.DataFrame): 정규화된 테스트 데이터
        y_train (pd.Series): 훈련 데이터의 타겟 변수
        y_test (pd.Series): 테스트 데이터의 타겟 변수
        train_data_sorted (pd.DataFrame): 정렬된 훈련 데이터
        test_data_sorted (pd.DataFrame): 정렬된 테스트 데이터
        landscape (int): 경관 요소 선택 옵션
    
    반환값:
        tuple: (학습된 최적 모델, 최적 하이퍼파라미터, 훈련 데이터 R², 테스트 데이터 R², 훈련 데이터 RMSE, 테스트 데이터 RMSE, 모델 이름)
    """
    
    np.random.seed(42)
    random.seed(42)
    rstate = np.random.default_rng(42) # 랜덤 시드 설정 (재현성 보장)

    def objective(params):
        """ 
        최적화 과정에서 사용되는 목적 함수 
        """
        for key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'num_leaves', 'min_child_samples']:
            if key in params:
                params[key] = int(params[key])

        model = model_cls(**params, random_state=42)   # 모델 인스턴스 생성

        # TimeSeriesSplit 적용 (3개 분할 사용)
        tscv = TimeSeriesSplit(n_splits=3)
        rmse_scores = []

        for train_idx, val_idx in tscv.split(X_train_scaled):
            X_train_fold, X_val_fold = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))

        avg_rmse = np.mean(rmse_scores)  # TimeSeriesSplit 기반 RMSE 계산
        return {'loss': avg_rmse, 'status': STATUS_OK}

    log_filename = f"{model_name}_training.log"
    log_filepath = os.path.join(save_path, log_filename)

    original_stdout = sys.stdout  # 기존 표준 출력을 저장

    try:
        os.makedirs(save_path, exist_ok=True)  
        sys.stdout = open(log_filepath, 'w', encoding="utf-8")
        sys.stderr = sys.stdout  

        trials = Trials()
        best_params = fmin(fn=objective, space=param_space, algo=tpe.suggest, 
                        max_evals=n_iter_count, trials=trials, rstate=rstate)  # 🔥 hyperopt 시드 고정

        # 🔥 최적 파라미터 변환 추가(float으로 반환돼서 파라미터가 0으로 간주되는 것 방지)
        for key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'num_leaves', 'min_child_samples']:
            if key in best_params:
                best_params[key] = int(best_params[key])

        best_model = model_cls(**best_params, random_state=42)  
        best_model.fit(X_train_scaled, y_train)  

        joblib.dump(best_model, os.path.join(save_path, f"{model_name}.pkl"))

        y_pred_train = best_model.predict(X_train_scaled)
        y_pred_test = best_model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"📊 Best Hyperparameters for {model_name}: {best_params}")
        print(f"✅ {model_name} RMSE (Train): {train_rmse:.4f}, RMSE (Test): {test_rmse:.4f}")
        print(f"✅ {model_name} R² (Train): {train_r2:.4f}, R² (Test): {test_r2:.4f}")

        print(f"🛠 DEBUG: {model_name} 모델 학습 완료, load_and_predict() 호출 시작", flush=True)

        try:
            load_and_predict(save_path, model_name, X_test_scaled, train_data_sorted, test_data_sorted, 
                            best_params, train_rmse, test_rmse, train_r2, test_r2, landscape, n_iter_count)
            print(f"🎯 {model_name} 예측 및 시각화 완료", flush=True)
        except Exception as e:
            print(f"⚠️ ERROR: {model_name} 예측 및 시각화에서 오류 발생: {e}", flush=True)

    finally:
        try:
            sys.stdout.flush()  # 로그 유실 방지
            sys.stdout.close()
        except Exception as e:
            print(f"⚠️ ERROR: 로그 파일 닫는 중 오류 발생: {e}", flush=True)
        
        sys.stdout = original_stdout  # 표준 출력 복원

    return best_model, best_params, train_r2, test_r2, train_rmse, test_rmse, model_name



def run_models(n_iter_count, save_path, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted, landscape):
    """ 
    여러 모델을 병렬로 학습시키고 결과를 저장하는 함수
    반환값:
        dict: 모델 이름을 키로 하는 학습 결과 (모델 객체, 하이퍼파라미터, 평가 지표 포함)
    """
    models = {
        'lgbm': (LGBMRegressor, get_param_space('lgbm')),
        'rf': (RandomForestRegressor, get_param_space('rf')),
        'gb': (GradientBoostingRegressor, get_param_space('gb')),
        'xgb': (XGBRegressor, get_param_space('xgb'))
    }

    models_to_run = {name: (cls, space) for name, (cls, space) in models.items() if name in config.MODELS_TO_RUN}
    max_cores = multiprocessing.cpu_count()
    optimal_cores = max(max_cores - 2, 1)  # 최소 1개 코어 유지

    pool = multiprocessing.Pool(processes=optimal_cores)
    results = {}

    def log_result(model_result):
        """ 
        개별 모델 학습이 완료될 때 호출되는 콜백 함수 
        매개변수:
            model_result (tuple): 학습된 모델의 결과 (튜플 형식)
        """
        if isinstance(model_result, tuple) and len(model_result) == 7:
            model_name = model_result[-1]  # 튜플의 마지막 값이 모델 이름
            results[model_name] = model_result
            print(f"✅ {model_name} 모델 학습 완료: {model_result[:3]}...")  # 일부만 출력하여 가독성 유지
        else:
            print(f"⚠️ 결과 형식이 예상과 다릅니다: {model_result}")

    for name, (cls, space) in models_to_run.items():
        pool.apply_async(
            train_model, 
            args=(cls, space, n_iter_count, save_path, name, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted, landscape),
            callback=log_result  # ✅ 모델 학습 완료 시 바로 결과 출력
        )

    pool.close()
    pool.join()

    return results



def get_unique_filename(directory, base_name, extension="png"):
    """ 
    저장된 그래프 파일명이 중복되지 않도록 자동으로 번호를 부여하는 함수
    
    매개변수:
        directory (str): 파일이 저장될 디렉토리 경로
        base_name (str): 파일의 기본 이름 (예: "model_mean_weather")
        extension (str): 파일 확장자 (기본값: "png")
    
    반환값:
        str: 고유한 파일 경로
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    existing_files = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(f".{extension}")]
    
    # 기존 파일에서 숫자 찾기
    numbers = []
    for f in existing_files:
        parts = f.split("_")
        num_part = parts[-1].split(".")[0]  # 확장자 앞 숫자 추출
        if num_part.isdigit():
            numbers.append(int(num_part))

    next_number = max(numbers) + 1 if numbers else 1  # 기존 숫자 중 가장 큰 값 +1, 없으면 1

    return os.path.join(directory, f"{base_name}_{next_number}.{extension}")

def load_and_predict(save_path, model_name, X_test_scaled, train_data_sorted, test_data_sorted, 
                     best_params, train_rmse, test_rmse, train_r2, test_r2, landscape, n_iter_count):
    """ 
    저장된 모델을 불러와 테스트 데이터에 대한 예측을 수행하고 결과를 시각화하는 함수
    """
    #  (월, 일) 단위 평균 데이터 계산 
    test_data_sorted["Month"] = test_data_sorted["DATE"].dt.month
    test_data_sorted["Day"] = test_data_sorted["DATE"].dt.day

    # 실제 모기 개체 수 평균 계산
    test_data_avg = test_data_sorted.groupby(["Month", "Day"])["mosquito"].mean().reset_index()
    test_data_avg["Date"] = pd.to_datetime(test_data_avg[["Month", "Day"]].assign(Year=2024))
    test_data_avg.set_index("Date", inplace=True)
    ''' 
    멀티프로세싱(multiprocessing) 환경에서 전역 변수를 공유하는 것이 복잡하므로, 각 모델 실행 시마다 test_data_avg를 생성하는 것이 더 나은 방식
    즉 run_models()에서 한 번만 실행하는 대신, train_model()에서 필요할 때마다 생성하도록 변경
    '''
    # 모델 풀네임 및 색상 정보 (함수 내부에서 정의)
    model_full_names = {
        "lgbm": "Light Gradient Boosting",
        "xgb": "Extreme Gradient Boosting",
        "rf": "Random Forest",
        "gb": "Gradient Boosting"
    }

    model_colors = {
        "lgbm": "green",
        "xgb": "red",
        "rf": "blue",
        "gb": "orange"
    }
    
    # 모델별 실행마다 초기화 
    predictions = {}  
    y_pred_avg = {}
    full_name = model_full_names.get(model_name, model_name)

    # 모델 불러오기
    model_path = os.path.join(save_path, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        print(f"⚠️ {model_full_names[model_name]} 모델 파일이 존재하지 않습니다: {model_path}")
        return  # ✅ 모델 파일이 없으면 함수 종료

    model = joblib.load(model_path)
    test_data_sorted[model_name] = model.predict(X_test_scaled)  # ✅ 모델별 컬럼에 저장

    # 예측값을 DataFrame으로 변환 및 월-일 단위 평균 계산
    y_pred_df = pd.DataFrame({"DATE": test_data_sorted["DATE"], model_name: test_data_sorted[model_name]})
    y_pred_df["Month"] = y_pred_df["DATE"].dt.month
    y_pred_df["Day"] = y_pred_df["DATE"].dt.day

    # 모델별 예측값 저장
    y_pred_avg[model_name] = y_pred_df.groupby(["Month", "Day"])[model_name].mean().reset_index()
    y_pred_avg[model_name].rename(columns={model_name: "Prediction"}, inplace=True)  # ✅ "Prediction" 컬럼 추가
    y_pred_avg[model_name]["Date"] = pd.to_datetime(y_pred_avg[model_name][["Month", "Day"]].assign(Year=2024))
    y_pred_avg[model_name].set_index("Date", inplace=True)

    print(f"📌 {model_name} 평균 예측 데이터 생성 완료")

    # 예측값 및 평가 지표 저장 (모델별 독립적 유지)
    predictions[model_name] = {
        "y_pred": test_data_sorted[model_name].values,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse
    }
    print(f"📌 {model_name} 예측값 및 평가 지표 저장 완료")

    # 그래프 출력 및 저장
    plt.figure(figsize=(16, 8))
    # 실제 모기 개체 수 평균값
    plt.plot(test_data_avg.index, test_data_avg['mosquito'], label="Observation",
             color="grey", marker='o', markersize=7, linewidth=2)
    # 모델별 예측값 시각화
    plt.plot(y_pred_avg[model_name].index, y_pred_avg[model_name]["Prediction"], 
             label="Prediction", color=model_colors[model_name], 
             marker='o', markersize=7, linewidth=2)
    # X축 포맷 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    # y축 회색 세로줄 제거 (x축만 그리드 유지)
    plt.grid(axis='x')
    # y축 범위 자동 조정 (최대값이 100 이하이면 0~100으로 설정, 초과하면 자동 조정)
    max_value = max(test_data_avg['mosquito'].max(), y_pred_avg[model_name]['Prediction'].max())
    if max_value > 100:
        plt.ylim(None)  # 자동 조정
    else:
        plt.ylim(0, 100)  # 기본값 설정  
    # 제목 수정: 선택한 모델의 풀네임이 나오도록 변경
    plt.title(f"{full_name}", fontsize=25)
    plt.legend(fontsize=17, loc='upper left')
    # R² 및 RMSE 정보 박스
    textstr = (f"R² (Train): {train_r2:.4f}  RMSE (Train): {train_rmse:.4f}\n"
               f"R² (Test): {test_r2:.4f}  RMSE (Test): {test_rmse:.4f}\n"
               f"Trial: {n_iter_count}  Landscape: {landscape}")
    plt.figtext(0.753, 0.853, textstr, fontsize=15, ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.5'))
    
    #  그래프 저장
    base_filename = f"{model_name}_mean_ld{landscape}"
    save_path = get_unique_filename(save_path, base_filename, "png")  # 중복 방지 적용
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📁 그래프 저장 완료: {save_path}")



## tasklist | findstr python

## taskkill /F /IM python3.11.exe
