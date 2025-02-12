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
    """config.py에서 하이퍼파라미터 가져와서 hyperopt 형태로 변환"""
    param_ranges = config.HYPERPARAMS[model_name]
    
    param_space = {}
    for key, value in param_ranges.items():
        if len(value) == 3:  # quniform (정수형)
            param_space[key] = hp.quniform(key, *value)
        elif len(value) == 2:  # loguniform (학습률 등)
            param_space[key] = hp.uniform(key, value[0], value[1])  # loguniform 대신 uniform 사용
    
    return param_space


def train_model(model_cls, param_space, n_iter_count, save_path, region, model_name, 
                X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, 
                test_data_sorted, landscape):
    # 🔹 각 프로세스에서 랜덤 시드 고정 (재현성 보장)
    np.random.seed(42)
    random.seed(42)
    rstate = np.random.default_rng(42)

    def objective(params):
        for key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'num_leaves', 'min_child_samples']:
            if key in params:
                params[key] = int(params[key])

        model = model_cls(**params, random_state=42)  

        # TimeSeriesSplit 적용 (3개 분할 사용)
        tscv = TimeSeriesSplit(n_splits=3)
        rmse_scores = []

        for train_idx, val_idx in tscv.split(X_train_scaled):
            X_train_fold, X_val_fold = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))

        avg_rmse = np.mean(rmse_scores)  # ✅ TimeSeriesSplit 기반 RMSE 계산
        return {'loss': avg_rmse, 'status': STATUS_OK}

    log_filename = f"{model_name}_training.log"
    log_filepath = os.path.join(save_path, log_filename)

    original_stdout = sys.stdout

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
        print(X_train_scaled.columns)

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

        load_and_predict(save_path, model_name, X_test_scaled, train_data_sorted, test_data_sorted, best_params, train_rmse, test_rmse, train_r2, test_r2, landscape, n_iter_count)

        print("🎯 {model_name} 예측 및 시각화 완료")

    finally:
        sys.stdout.flush()  # 로그 유실 방지
        sys.stdout.close()
        sys.stdout = original_stdout  

    return best_model, best_params, train_r2, test_r2, train_rmse, test_rmse, model_name



def run_models(n_iter_count, save_path, region, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted, landscape):
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
        """ 모델 학습 완료 후 즉시 결과를 출력하는 함수 """
        if isinstance(model_result, tuple) and len(model_result) == 7:
            model_name = model_result[-1]  # 튜플의 마지막 값이 모델 이름
            results[model_name] = model_result
            print(f"✅ {model_name} 모델 학습 완료: {model_result[:3]}...")  # 일부만 출력하여 가독성 유지
        else:
            print(f"⚠️ 결과 형식이 예상과 다릅니다: {model_result}")

    for name, (cls, space) in models_to_run.items():
        pool.apply_async(
            train_model, 
            args=(cls, space, n_iter_count, save_path, region, name, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted, landscape),
            callback=log_result  # ✅ 모델 학습 완료 시 바로 결과 출력
        )

    pool.close()
    pool.join()

    return results



def get_unique_filename(directory, base_name, extension="png"):
    """
    저장된 그래프 파일명이 중복되지 않도록 자동 번호를 매김.
    예: {model_name}_mean_weather_Ld{landscape}_1.png, _2.png, _3.png ...
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

    # 기존 model.py 방식대로 모델 불러오기
    model = joblib.load(os.path.join(save_path, f"{model_name}.pkl"))
    test_data_sorted["Prediction"] = model.predict(X_test_scaled)

    # (월, 일) 단위 평균 데이터 계산
    test_data_sorted["Month"] = test_data_sorted["DATE"].dt.month
    test_data_sorted["Day"] = test_data_sorted["DATE"].dt.day

    # 실제 모기 개체 수 (Observation) 평균 계산
    test_data_avg = test_data_sorted.groupby(["Month", "Day"])["mosquito"].mean().reset_index()
    test_data_avg["Date"] = pd.to_datetime(test_data_avg[["Month", "Day"]].assign(Year=2024))
    test_data_avg.set_index("Date", inplace=True)

    # 모델별 예측값 평균 계산
    y_pred_avg = test_data_sorted.groupby(["Month", "Day"])["Prediction"].mean().reset_index()
    y_pred_avg["Date"] = pd.to_datetime(y_pred_avg[["Month", "Day"]].assign(Year=2024))
    y_pred_avg.set_index("Date", inplace=True)

    print(f"📌 {model_name} 평균 예측 데이터 생성 완료")

    # 평균 그래프 출력 및 저장
    plt.figure(figsize=(16, 8))
    # 실제 모기 개체 수 평균값 (회색, 선 두께 증가)
    plt.plot(test_data_avg.index, test_data_avg['mosquito'], label="Observation",
             color="grey", marker='o', markersize=7, linewidth=2)
    # 예측값 평균 (선택한 모델의 색상 적용)
    plt.plot(y_pred_avg.index, y_pred_avg['Prediction'], 
             label="Prediction", color=model_colors[model_name], marker='o', markersize=7, linewidth=2)
    # X축 포맷 설정 (월-일)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    # y축 회색 세로줄 제거 (x축만 그리드 유지)
    plt.grid(axis='x')
    # 폰트 크기 설정
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # y축 범위 자동 조정 (최대값이 100 이하이면 0~100으로 설정, 초과하면 자동 조정)
    max_value = max(test_data_avg['mosquito'].max(), y_pred_avg[model_name]['Prediction'].max())
    if max_value > 100:
        plt.ylim(None)  # 자동 조정
    else:
        plt.ylim(0, 100)  # 기본값 설정
    # 제목 수정: 선택한 모델의 풀네임이 나오도록 변경
    plt.title(f"{model_full_names[model_name]}", fontsize=25)
    # 범례 왼쪽 상단으로 이동
    plt.legend(fontsize=17, loc='upper left')
    # R² 및 RMSE 정보 텍스트 박스
    textstr = (f"R² (Train): {train_r2:.4f}  RMSE (Train): {train_rmse:.4f}\n"
               f"R² (Test): {test_r2:.4f}  RMSE (Test): {test_rmse:.4f}\n"
               f"Trial: {n_iter_count}  Landscape: {landscape}")

    plt.figtext(0.753, 0.853, textstr, fontsize=15, ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round,pad=0.5'))

    #  기존 방식대로 그래프 저장하되, 자동 번호 부여
    base_filename = f"{model_name}_mean_weather_Ld{landscape}"
    unique_file_path = get_unique_filename(save_path, base_filename, "png")

    # 그래프 저장
    plt.savefig(save_path, dpi=300,bbox_inches='tight')
    print(f"📁 그래프 저장 완료: {unique_file_path}")



## tasklist | findstr python

## taskkill /F /IM python3.11.exe
