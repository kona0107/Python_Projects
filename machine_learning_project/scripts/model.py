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


def get_param_space(model_name):
    """config.py에서 하이퍼파라미터 가져와서 hyperopt 형태로 변환"""
    param_ranges = config.HYPERPARAMS[model_name]
    
    param_space = {}
    for key, value in param_ranges.items():
        if len(value) == 3:  # quniform
            param_space[key] = hp.quniform(key, *value)
        elif len(value) == 2:  # loguniform (learning_rate 같은 경우)
            param_space[key] = hp.loguniform(key, np.log(max(value[0], 1e-3)), np.log(value[1]))
    
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

        model = model_cls(**params, random_state=42)  # ✅ 모든 CPU 사용

        # ✅ TimeSeriesSplit 적용 (3개 분할 사용)
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

        # 🔥 최적 파라미터 변환 추가
        for key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'num_leaves', 'min_child_samples']:
            if key in best_params:
                best_params[key] = int(best_params[key])

        best_model = model_cls(**best_params, random_state=42)  
        if model_name in ["xgb", "lgbm"]:
            best_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                early_stopping_rounds=50,  # ✅ 성능 개선이 없으면 학습 중단
                verbose=100
            )
        else:
            best_model.fit(X_train_scaled, y_train)  # ❌ eval_set 없이 학습


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

    finally:
        sys.stdout.flush()  # ✅ 로그 유실 방지
        sys.stdout.close()
        sys.stdout = original_stdout  

    return best_model


def run_models(n_iter_count, save_path, region, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted, landscape):
    models = {
        'lgbm': (LGBMRegressor, get_param_space('lgbm')),
        'rf': (RandomForestRegressor, get_param_space('rf')),
        'gb': (GradientBoostingRegressor, get_param_space('gb')),
        'xgb': (XGBRegressor, get_param_space('xgb'))
    }
    
    models_to_run = {name: (cls, space) for name, (cls, space) in models.items() if name in config.MODELS_TO_RUN}
    max_cores = multiprocessing.cpu_count()
    optimal_cores = max(2, max_cores - 2)  # ✅ 최소 2개 코어 유지

    pool = multiprocessing.Pool(processes=optimal_cores)
    results = {}
    async_results = []

    for name, (cls, space) in models_to_run.items():
        async_result = pool.apply_async(
            train_model, 
            args=(cls, space, n_iter_count, save_path, region, name, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted, landscape)
        )
        async_results.append((name, async_result))

    pool.close()
    pool.join()

    for name, async_result in async_results:
        results[name] = async_result.get()

    return results


def get_unique_filename(directory, base_name, landscape, extension="png"):
    """
    주어진 디렉토리에서 base_name과 landscape를 포함한 파일명을 찾아
    가장 큰 숫자 다음 숫자로 파일명 생성
    """
    if not os.path.exists(directory):  # 디렉토리 존재 여부 확인 후 생성
        os.makedirs(directory, exist_ok=True)

    filename_prefix = f"{base_name}_landscape{landscape}_"
    existing_files = [f for f in os.listdir(directory) if f.startswith(filename_prefix) and f.endswith(f".{extension}")]

    # 정규 표현식을 사용하여 파일명에서 숫자 추출 (안정성 개선)
    numbers = []
    pattern = re.compile(rf"{re.escape(filename_prefix)}(\d+)\.{extension}")
    
    for f in existing_files:
        match = pattern.match(f)
        if match:
            numbers.append(int(match.group(1)))

    next_number = max(numbers) + 1 if numbers else 1  # 숫자 없을 경우 기본값 1

    return os.path.join(directory, f"{filename_prefix}{next_number}.{extension}")


def load_and_predict(save_path, model_name, X_test_scaled, train_data_sorted, test_data_sorted, best_params, train_rmse, test_rmse, train_r2, test_r2, landscape, n_iter_count):
    model_full_names = {
        "lgbm": "LightGBM",
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
    
    full_model_name = model_full_names.get(model_name, model_name)
    model_color = model_colors.get(model_name, "red")
    model = joblib.load(os.path.join(save_path, f"{model_name}.pkl"))
    y_test_pred = model.predict(X_test_scaled)
    
    plt.figure(figsize=(12, 8))
    plt.plot(test_data_sorted['DATE'], test_data_sorted['mosquito'], label='Actual', color='black')
    plt.plot(test_data_sorted['DATE'], y_test_pred, label='Prediction', color=model_color)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Mosquito Count")
    plt.title(f"{full_model_name} Prediction", fontsize=25)
    plt.grid()
    
    textstr = f"RMSE (Train): {train_rmse:.4f}  RMSE (Test): {test_rmse:.4f}\nR² (Train): {train_r2:.4f}  R² (Test): {test_r2:.4f}\nLandscape: {landscape}  n_iter_count: {n_iter_count}"
    plt.figtext(0.5, 0.08, textstr, fontsize=14, ha='center', va='center')
    plt.subplots_adjust(bottom=0.25)
    
    os.makedirs(save_path, exist_ok=True)
    unique_filename = get_unique_filename(save_path, model_name, landscape, "png")  # ✅ 파일명 자동 증가
    plt.savefig(unique_filename)
    print(f"✅ 그래프 저장 완료: {unique_filename}")


## tasklist | findstr python

## taskkill /F /IM python3.11.exe
