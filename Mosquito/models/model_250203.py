import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import matplotlib.dates as mdates
from glob import glob
from math import sqrt

# Scikit-learn 관련 라이브러리
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# 추가적인 머신러닝 모델 및 최적화 라이브러리
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from skopt import BayesSearchCV -> hyperopt로 변경


def data_preprocessing(path):
    # 데이터 로드
    df = pd.read_excel(path, engine='openpyxl')

    # IQR 계산
    Q1_Mosquitoe = df['mosquito'].quantile(0.25)
    Q3_Mosquitoe = df['mosquito'].quantile(0.75)
    IQR_Mosquitoe = Q3_Mosquitoe - Q1_Mosquitoe
    lower_bound_Mosquitoe = Q1_Mosquitoe - 1.5 * IQR_Mosquitoe
    upper_bound_Mosquitoe = Q3_Mosquitoe + 1.5 * IQR_Mosquitoe

    # 이상치 제거
    df_iqr = df[(df['mosquito'] >= lower_bound_Mosquitoe) & (df['mosquito'] <= upper_bound_Mosquitoe)]

    # Train-Test 분리
    train = df_iqr[df_iqr['DATE'] <= '2023-10-31']
    test = df_iqr[df_iqr['DATE'] >= '2024-04-01']

    # 날짜(datetime64) 컬럼 제거
    if 'DATE' in train.columns:
        train = train.drop(columns=['DATE'])
        test = test.drop(columns=['DATE'])

    # 범주형(object) 데이터 제거
    train = train.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])

    # MinMaxScaler 적용
    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)

    # 독립 변수(X)와 종속 변수(y) 분리
    target_col = 'mosquito'
    X_train = train_scaled.drop(columns=[target_col])
    y_train = train_scaled[target_col]
    X_test = test_scaled.drop(columns=[target_col])
    y_test = test_scaled[target_col]

    # 데이터 크기 확인
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 데이터 반환
    return X_train, y_train, X_test, y_test, train, test

# 공통 모델 학습 함수 (hyperopt 적용)
def train_model(model_cls, param_space, path, n_iter_count, save_path, region, model_name):
    
    # 데이터 불러오기
    X_train, y_train, X_test, y_test, train, test = data_preprocessing(path)

    def objective(params):
        """ 최적화할 목적 함수 """
        # 🔥 float -> int 변환 (중요)
        for key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'num_leaves', 'min_child_samples']:
            if key in params:
                params[key] = int(params[key])


        model = model_cls(**params)  # 모델 초기화
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 모델 평가 (RMSE 계산)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return {'loss': rmse, 'status': STATUS_OK}

    # Hyperparameter Optimization 실행
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,  # Bayesian Optimization (TPE)
        max_evals=n_iter_count,
        trials=trials
    )

    # 🔥 최적 파라미터 변환 추가 (반드시 fmin() 이후에 위치)
    for key in ['n_estimators','max_depth', 'min_samples_split', 'min_samples_leaf', 'num_leaves', 'min_child_samples']:
        if key in best_params:
            best_params[key] = int(best_params[key])

    # 최적 모델 학습
    best_model = model_cls(**best_params)
    best_model.fit(X_train, y_train)

    # 최적 모델 저장
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(best_model, os.path.join(save_path, f"{model_name}.pkl"))

    # 성능 평가
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"📊 Best Hyperparameters for {model_name}: {best_params}")
    print(f"✅ {model_name} RMSE (Train): {train_rmse:.4f}, RMSE (Test): {test_rmse:.4f}")
    print(f"✅ {model_name} R² (Train): {train_r2:.4f}, R² (Test): {test_r2:.4f}")

    return best_params, best_model, train_r2, test_r2, train_rmse, test_rmse, model_name


# 모델 실행 함수 (hyperopt 적용)
def run_models(path, n_iter_count, save_path, region):

    models = {
        "lgbm": (LGBMRegressor, {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
            'max_depth': hp.quniform('max_depth', 1, 200, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1.0)),
            'num_leaves': hp.quniform('num_leaves', 2, 100, 1),
            'min_child_samples': hp.quniform('min_child_samples', 1, 50, 1)
        }),
        "rf": (RandomForestRegressor, {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
            'max_depth': hp.quniform('max_depth', 1, 200, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1)
        }),
        "gb": (GradientBoostingRegressor, {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
            'max_depth': hp.quniform('max_depth', 1, 200, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1.0))
        }),
        "xgb": (XGBRegressor, {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
            'max_depth': hp.quniform('max_depth', 1, 200, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1.0)),
            'subsample': hp.uniform('subsample', 0.1, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1.0)
        })
    }

    results = {}
    for model_name, (model_cls, param_space) in models.items():
        results[model_name] = train_model(model_cls, param_space, path, n_iter_count, save_path, region, model_name)

    return results


# 모델 로드 및 예측 함수(임시)
def load_and_predict(path, save_path, model_name):
    """저장된 모델을 로드하고 예측 수행 후 시각화"""
    _, _, _, _, train, test = data_preprocessing(path)
    model = joblib.load(os.path.join(save_path, f"{model_name}.pkl"))
    
    y_test_pred = model.predict(test.drop(columns=['DATE', 'mosquito']))
    
    # 예측 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(test['DATE'], test['mosquito'], label='Actual', color='black')
    plt.plot(test['DATE'], y_test_pred, label='Predicted', color='red')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Mosquito Count")
    plt.title(f"{model_name} Prediction")
    plt.grid()
    plt.show()

# 실행 예시
path = r'/home/kona0107/EDAM/2015_2024_total.xlsx'
save_path = r'/home/kona0107/EDAM/models'
n_iter_count = 1
region = "Seoul"

results = run_models(path, n_iter_count, save_path, region)

