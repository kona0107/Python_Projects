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
    filtered_data_group = filtered_data.groupby('DMS_CODE')
    for i in range(1, 22) :
        dataset[f'mosquito_rolling{i}'] = filtered_data_group['mosquito'].shift(i)
    dataset.dropna(inplace=True)
    
    dataset.drop('landscape', axis=1, inplace=True)

    # 연도를 기준으로 데이터 분리하기
    dataset.loc[:, 'Year'] = pd.to_datetime(dataset['DATE']).dt.year  # 명시적으로 Year 열 추가
    
    TEST_YEAR = config.TEST_YEAR
    # 훈련 데이터: 2015년부터 2022년까지
    train_data = dataset[dataset['Year'] < TEST_YEAR]
    train_data_sorted = train_data.sort_values('DATE')

    # 테스트 데이터: 2023년 이후
    test_data = dataset[dataset['Year'] >= TEST_YEAR]
    test_data_sorted = test_data.sort_values('DATE')
    
    X_train = train_data_sorted.drop('mosquito', axis=1)
    X_test = test_data_sorted.drop('mosquito', axis=1)
    y_train = train_data_sorted['mosquito']
    y_test = test_data_sorted['mosquito']
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train.iloc[:, 4:-1]), columns= X_train.iloc[:, 4:-1].columns, index=X_train.iloc[:, 4:-1].index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test.iloc[:, 4:-1]), columns= X_test.iloc[:, 4:-1].columns, index=X_test.iloc[:, 4:-1].index)
    
    return scaler, X_train_scaled, X_test_scaled, y_train, y_test, X_train, train_data_sorted, test_data_sorted

if __name__ == "__main__" :
    data = pd.read_excel("F:/박정현/ML/machine_learning_project/data/weather/2015_2023_daily_data.xlsx")
    scaler, X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = processed_data(data)

