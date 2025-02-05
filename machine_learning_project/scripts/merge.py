import pandas as pd
import config
import glob
import os

def merge_data(weather):
    envi = pd.read_csv(config.BASE_DIR + "/machine_learning_project/data/env/seoul_240930.csv", encoding='cp949')
    # 폴더 경로 설정 
    folder_path = config.MOSQUITO_PATH

    # DMS로 시작하는 엑셀 파일 목록
    files = glob.glob(os.path.join(folder_path, 'DMS*.xlsx'))

    # 모든 엑셀 파일을 읽고 concat으로 합치기
    df_list = []
    for file in files:
        df = pd.read_excel(file)
        df_list.append(df)

    # DataFrame 합치기
    data = pd.concat(df_list, ignore_index=True)
    envi = envi.loc[:, ['측정소명', 'DMS_open_1m', 'DMS_open_3m', 'DMS_open_5m', 'DMS_open_10m', 'Env_Factors_1m', 'Env_Factors_3m', 'Env_Factors_5m', 'Env_Factors_10m', '경관요소']]
    envi.drop_duplicates(inplace=True)
    
    envi.dropna(inplace=True)
    merge = pd.merge(weather, envi, how='left', on =['측정소명'] )
    merge.dropna(inplace=True)
    data = data[data['측정일'] != '합계']
    data = data.iloc[:, [0, 3, 5]]
    mosquito = data.groupby(['측정일', '측정소명'], as_index= False).sum()
    weather['측정일'] = pd.to_datetime(weather['측정일'])
    mosquito['측정일'] = pd.to_datetime(mosquito['측정일'], errors='coerce')
    last = pd.merge(merge, mosquito, on=['측정일', '측정소명'], how ='left')
    last.drop('remark', axis =1, inplace= True)
    last.dropna(inplace=True)
    last = last.rename(columns= {"측정일" : "DATE", "측정소명" : "std_name", "모기" : "mosquito", '경관요소' : 'landscape'})
    last.to_excel(config.BASE_DIR +"/machine_learning_project/data/weather/2015_2023_total.xlsx", index=False)
    
    return last
    
# 모듈 테스트용 코드
if __name__ == "__main__":
    weather = pd.read_excel(config.BASE_DIR +"/machine_learning_project/data/weather/2015_2023_mosquito.xlsx")
    merge_data(weather)
    print("데이터 병합 완료")