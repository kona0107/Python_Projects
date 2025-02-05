import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import multiprocessing
import config
import sys
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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

def train_model(model_cls, param_space, n_iter_count, save_path, region, model_name, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted, landscape):
    def objective(params):
        for key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'num_leaves', 'min_child_samples']:
            if key in params:
                params[key] = int(params[key])
        
        model = model_cls(**params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        return {'loss': np.sqrt(mean_squared_error(y_test, y_pred)), 'status': STATUS_OK}
    
    log_filename = f"{model_name}_training.log"
    log_filepath = os.path.join(save_path, log_filename)
    
    original_stdout = sys.stdout  # ✅ 기존 stdout 저장
    try:
        sys.stdout = open(log_filepath, 'w', encoding="utf-8")
        sys.stderr = sys.stdout  

        trials = Trials()
        best_params = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=n_iter_count, trials=trials)
        
        # 🔥 최적 파라미터 변환 추가 (반드시 fmin() 이후에 위치)
        for key in ['n_estimators','max_depth', 'min_samples_split', 'min_samples_leaf', 'num_leaves', 'min_child_samples']:
            if key in best_params:
                best_params[key] = int(best_params[key])
        
        best_model = model_cls(**best_params)
        best_model.fit(X_train_scaled, y_train)
        
        os.makedirs(save_path, exist_ok=True)
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
    
    finally:
        sys.stdout.close()  # ✅ 파일 닫기
        sys.stdout = original_stdout  # ✅ 기존 stdout 복원

    return best_model



def run_models(n_iter_count, save_path, region, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted, landscape):
    models = {
        'lgbm': (LGBMRegressor, get_param_space('lgbm')),
        'rf': (RandomForestRegressor, get_param_space('rf')),
        'gb': (GradientBoostingRegressor, get_param_space('gb')),
        'xgb': (XGBRegressor, get_param_space('xgb'))
    }
    
    models_to_run = {name: (cls, space) for name, (cls, space) in models.items() if name in config.MODELS_TO_RUN}
    
    max_cores = multiprocessing.cpu_count()  # 시스템의 최대 CPU 코어 수 확인
    optimal_cores = max(2, max_cores - 2)  # 전체 코어 - 2개 사용 (안정성 고려)

    pool = multiprocessing.Pool(processes=optimal_cores)
    results = {}
    async_results = []

    # apply_async()에서 args= 추가하여 명확하게 인자 전달
    for name, (cls, space) in models_to_run.items():
        async_result = pool.apply_async(
            train_model, 
            args=(cls, space, n_iter_count, save_path, region, name, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted, landscape)
        )

        async_results.append((name, async_result))

    # 모든 작업이 등록된 후 close() 및 join()
    pool.close()
    pool.join()
    pool.terminate() 

    for name, async_result in async_results:
        results[name] = async_result.get()

    return results


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
    
    plt.figure(figsize=(12, 8))  # ✅ 그래프 크기 확장
    plt.plot(test_data_sorted['DATE'], test_data_sorted['mosquito'], label='Actual', color='black')
    plt.plot(test_data_sorted['DATE'], y_test_pred, label='Prediction', color=model_color)  # ✅ 모델별 색상 지정
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Mosquito Count")
    plt.title(f"{full_model_name} Prediction", fontsize=20)  # ✅ 제목을 풀네임으로 변경하고 글자 크기 확대
    plt.grid()
    
   # RMSE, R² 값 및 추가 정보 (Landscape, n_iter_count) 그래프 하단에 텍스트로 추가 (위치 및 크기 조정)
    textstr = (f"RMSE (Train): {train_rmse:.4f}  RMSE (Test): {test_rmse:.4f}\n"
           f"R² (Train): {train_r2:.4f}  R² (Test): {test_r2:.4f}\n"
           f"Landscape: {landscape}  n_iter_count: {n_iter_count}")
    plt.figtext(0.5, 0.05, textstr, fontsize=10, ha='center', va='center')
    plt.subplots_adjust(bottom=0.2)
    
    if multiprocessing.current_process().name == "MainProcess":
        plt.show(block=False)  # ✅ GUI 창이 떠도 메인 프로세스가 멈추지 않도록 변경
    else:
        plt.savefig(os.path.join(save_path, f"{model_name}.png"))  # ✅ 서브 프로세스에서는 자동 저장
    
    return y_test_pred


## tasklist | findstr python

## taskkill /F /IM python3.11.exe
