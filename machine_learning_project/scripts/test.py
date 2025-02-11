import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# ✅ 모델 결과를 시각화하는 함수
def visualize_model_predictions(model_name, save_path, X_test_scaled, test_data_sorted, landscape):
    # ✅ 저장된 모델 불러오기
    model = joblib.load(os.path.join(save_path, f"{model_name}.pkl"))

    # ✅ 예측 수행
    y_test_pred = model.predict(X_test_scaled)

    # ✅ 모델 이름 매핑
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

    # ✅ 그래프 생성
    plt.figure(figsize=(12, 8))
    plt.plot(test_data_sorted['DATE'], test_data_sorted['mosquito'], label='Actual', color='black')
    plt.plot(test_data_sorted['DATE'], y_test_pred, label='Prediction', color=model_color)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Mosquito Count")
    plt.title(f"{full_model_name} Prediction", fontsize=25)
    plt.grid()

    # ✅ 그래프 저장 경로 설정
    plots_dir = os.path.join(save_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    unique_filename = os.path.join(plots_dir, f"{model_name}_landscape{landscape}.png")
    plt.savefig(unique_filename)
    print(f"✅ 그래프 저장 완료: {unique_filename}")

# ✅ 실행 코드
if __name__ == "__main__":
    # ✅ 저장된 모델 경로
    save_path = r'F:/박정현/ML/machine_learning_project/models'

    # ✅ 데이터 불러오기 (테스트 데이터)
    test_data_path = r'F:/박정현/ML/machine_learning_project/data/processed/test_data.csv'
    X_test_scaled_path = r'F:/박정현/ML/machine_learning_project/data/processed/X_test_scaled.csv'

    # ✅ 테스트 데이터 로드
    test_data_sorted = pd.read_csv(test_data_path, parse_dates=["DATE"])
    X_test_scaled = pd.read_csv(X_test_scaled_path)

    # ✅ 실행할 모델 리스트
    model_list = ["rf", "gb"]

    for model_name in model_list:
        visualize_model_predictions(model_name, save_path, X_test_scaled, test_data_sorted, landscape=1)
