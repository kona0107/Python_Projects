import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import config

def processed_data(last):
    
    # Remove outliers in '모기' column using 1.5 * IQR
    Q1_Mosquitoe = last['mosquito'].quantile(0.25)
    Q3_Mosquitoe = last['mosquito'].quantile(0.75)
    IQR_Mosquitoe = Q3_Mosquitoe - Q1_Mosquitoe
    lower_bound_Mosquitoe = Q1_Mosquitoe - 1.5 * IQR_Mosquitoe
    upper_bound_Mosquitoe = Q3_Mosquitoe + 1.5 * IQR_Mosquitoe
    filtered_data = last[(last['mosquito'] >= lower_bound_Mosquitoe) & (last['mosquito'] <= upper_bound_Mosquitoe)]
    dataset = filtered_data.copy()

    # GroupBy 적용 (DMS_CODE 기준)
    filtered_data_group = filtered_data.groupby('DMS_CODE')

    # ✅✅✅ 롤링 피처 추가함(original 파일과의 차이점)
    for i in range(1, 22):  # 1~21일치 Rolling 적용 (모기 개체수)
        dataset[f'mosquito_rolling{i}'] = filtered_data_group['mosquito'].shift(i)

    for i in range(1, 8):  # 1~7일치 Rolling 적용 (기온, 강수량)
        dataset[f'TMAX_rolling{i}'] = filtered_data_group['TMAX'].shift(i)
        dataset[f'TMIN_rolling{i}'] = filtered_data_group['TMIN'].shift(i)
        dataset[f'PCP_rolling{i}'] = filtered_data_group['PCP'].shift(i)

    # 결측값 제거 (Rolling 적용 후 NaN 값 발생 가능)
    dataset.dropna(inplace=True)

    # 연도를 기준으로 데이터 분리
    dataset['Year'] = pd.to_datetime(dataset['DATE']).dt.year  # 명시적으로 Year 열 추가
    
    TEST_YEAR = config.TEST_YEAR
    # 훈련 데이터: 2015년부터 2023년까지
    train_data = dataset[dataset['Year'] < TEST_YEAR]
    train_data_sorted = train_data.sort_values('DATE')

    # 테스트 데이터: 2024년 이후
    test_data = dataset[dataset['Year'] >= TEST_YEAR]
    test_data_sorted = test_data.sort_values('DATE')
    
    # 독립변수(X)와 종속변수(y) 분리
    X_train = train_data_sorted.drop(columns=['mosquito'])
    X_test = test_data_sorted.drop(columns=['mosquito'])
    y_train = train_data_sorted['mosquito']
    y_test = test_data_sorted['mosquito']

    # ✅ MinMaxScaler 적용 (Rolling 피처 포함하여 정규화)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train.iloc[:, 4:]), 
                                  columns=X_train.iloc[:, 4:].columns, 
                                  index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test.iloc[:, 4:]), 
                                 columns=X_test.iloc[:, 4:].columns, 
                                 index=X_test.index)

    # ✅ Rolling feature가 정상적으로 추가됐는지 출력 확인
    print(f"✅ Rolling Features 추가 완료! 총 Feature 개수: {len(X_train.columns)}")
    # print("🔹 X_train 샘플 데이터:")
    # print(X_train.head())  터미널 가독성 위해 삭제

    return scaler, X_train_scaled, X_test_scaled, y_train, y_test, X_train, train_data_sorted, test_data_sorted


