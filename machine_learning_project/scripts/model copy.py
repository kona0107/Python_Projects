import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import concurrent.futures
import config
import sys
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

def get_param_space(model_name):
    """config.py에서 하이퍼파라미터 가져와서 hyperopt 형태로 변환"""
    param_ranges = config.HYPERPARAMS[model_name]
    
    param_space = {}
    for key, value in param_ranges.items():
        if len(value) == 3:  # quniform
            param_space[key] = hp.quniform(key, *value)
        elif len(value) == 2:  # uniform (learning_rate 같은 경우)
            param_space[key] = hp.loguniform(key, np.log(value[0]), np.log(value[1]))
    
    return param_space

from sklearn.model_selection import TimeSeriesSplit

def train_model(model_cls, param_space, n_iter_count, save_path, region, model_name, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted):
    def objective(params):
        for key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'num_leaves', 'min_child_samples']:
            if key in params:
                params[key] = int(params[key])

        model = model_cls(**params)

        # TimeSeriesSplit 추가
        tscv = TimeSeriesSplit(n_splits=5)
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
    
    sys.stdout = open(log_filepath, 'w')
    sys.stderr = sys.stdout  

    trials = Trials()
    best_params = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=n_iter_count, trials=trials)
    
    for key in best_params:
        best_params[key] = int(best_params[key])

    best_model = model_cls(**best_params)
    best_model.fit(X_train_scaled, y_train) 
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(best_model, os.path.join(save_path, f"{model_name}.pkl"))
    
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)
    
    train_r2, test_r2 = r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)
    train_rmse, test_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train)), np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"📊 Best Hyperparameters for {model_name}: {best_params}")
    print(f"✅ {model_name} RMSE (Train): {train_rmse:.4f}, RMSE (Test): {test_rmse:.4f}")
    print(f"✅ {model_name} R² (Train): {train_r2:.4f}, R² (Test): {test_r2:.4f}")
    
    load_and_predict(save_path, model_name, X_test_scaled, train_data_sorted, test_data_sorted)

    sys.stdout = sys.__stdout__
    return best_params, best_model, train_r2, test_r2, train_rmse, test_rmse, model_name


def run_models(n_iter_count, save_path, region, X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted):
    models_to_run = config.MODELS_TO_RUN  # config.py에서 실행할 모델 가져오기
    models = {
        'lgbm': (LGBMRegressor, get_param_space('lgbm')),
        'rf': (RandomForestRegressor, get_param_space('rf')),
        'gb': (GradientBoostingRegressor, get_param_space('gb')),
        'xgb': (XGBRegressor, get_param_space('xgb'))
    }

    # 선택된 모델만 실행하도록 필터링
    models = {name: (cls, space) for name, (cls, space) in models.items() if name in models_to_run}

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # 모델 학습을 위한 인수들을 넘기도록 수정
        future_to_model = {executor.submit(train_model, cls, space, n_iter_count, save_path, region, name,
                                           X_train_scaled, X_test_scaled, y_train, y_test, train_data_sorted, test_data_sorted): name
                           for name, (cls, space) in models.items()}
        
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                results[model_name] = future.result()
            except Exception as exc:
                print(f"⚠️ {model_name} 모델 학습 중 오류 발생: {exc}")
    
    return results

# 모델 로드 및 예측 함수(임시)
def load_and_predict(save_path, model_name, X_test_scaled, train_data_sorted, test_data_sorted):
    """저장된 모델을 로드하고 예측 수행 후 시각화"""
    model = joblib.load(os.path.join(save_path, f"{model_name}.pkl"))
    
    y_test_pred = model.predict(X_test_scaled)
    
    # 예측 결과 시각화 
    plt.figure(figsize=(10, 5))
    plt.plot(test_data_sorted['DATE'], test_data_sorted['mosquito'], label='Actual', color='black')
    plt.plot(test_data_sorted['DATE'], y_test_pred, label='Predicted', color='blue')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Mosquito Count")
    plt.title(f"{model_name} Prediction")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # 실행 예시
    save_path = r'/home/kona0107/EDAM/models'
    n_iter_count = 1  # 예시 반복 횟수
    region = "Seoul"

    # 모델 학습
    results = run_models(n_iter_count, save_path, region)
