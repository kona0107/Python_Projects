import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# 엑셀 파일 읽기
file_path = 'F:\박정현\ML\Mosquito\DMS_관측소_매핑.xlsx'
df = pd.read_excel(file_path)

# DMS CODE와 기상 관측소 코드 매핑 생성
dms_station_mapping = {}

for _, row in df.iterrows():
    dms_code = row['DMS CODE']


    
    dms_name = row['DMS 관측소명']
    first_priority = row['1순위 관측소 코드']
    second_priority = row['2순위 관측소 코드']
    
    dms_station_mapping[dms_code] = {
        '기상 1순위': first_priority,
        '기상 2순위': second_priority,
        'dms_name' : dms_name
    }

# 원하는 DMS CODE 목록
desired_dms_codes = df['DMS CODE'].tolist()

# API 설정
domain = "https://apihub.kma.go.kr/api/typ01/url/awsh.php?"
var = 'var='
api_keys = ["s5DFRYyOQqKQxUWMjlKi9g", "8CMfLj-MS2SjHy4_jAtkQQ"]  # 인증키 리스트
current_key_index = 0  # 현재 사용 중인 API 키 인덱스
api_call_count = 0  # 호출 횟수

def switch_api_key():
    # API 호출 횟수가 12,000번을 초과하면 키를 변경.
    global current_key_index
    current_key_index = (current_key_index + 1) % len(api_keys)
    print(f"Switching to API key: {api_keys[current_key_index]}")

def fetch_data(start_date, end_date):
    # 모든 데이터를 한 번에 가져오는 함수.
    global api_call_count
    data_for_all = []
    current_time = start_date

    while current_time <= end_date:
        # 19시부터 5시까지만 데이터 접근
        if current_time.hour >= 19 or current_time.hour <= 5:
            tm = current_time.strftime("%Y%m%d%H%M")
            url = f"{domain}{var}&tm={tm}&stn=0&authKey={api_keys[current_key_index]}"
            
            print(f"Fetching data for {current_time.strftime('%Y-%m-%d %H:%M')} with URL: {url}")
           
            try:
                response = requests.get(url)
                api_call_count += 1
                # 호출 횟수 확인 후 키 변경
                if api_call_count > 12000:
                    switch_api_key()
                    api_call_count = 0
                if response.status_code == 200:
                    text_data = response.text.strip()
                    if text_data:
                        lines = text_data.splitlines()
                        for line in lines:
                            if line.strip() and not line.startswith('#'):
                                fields = line.split()
                                station_id = int(fields[1])
                                # 원하는 기상 관측소 코드만 필터링
                                for dms_code in desired_dms_codes:
                                    first_priority = dms_station_mapping[dms_code]['기상 1순위']
                                    second_priority = dms_station_mapping[dms_code]['기상 2순위']
                                    dms_name = dms_station_mapping[dms_code]['dms_name']
                                    if station_id == first_priority or station_id == second_priority:
                                        temp = np.nan if float(fields[2]) == -99 else float(fields[2])
                                        wdspeed = np.nan if float(fields[4]) == -99 else float(fields[4])
                                        rain_1hr = np.nan if float(fields[6]) == -99 else float(fields[6])
                                        humidity = np.nan if float(fields[7]) == -99 else float(fields[7])
                                        # 'remark' 열을 1순위 또는 2순위로 설정
                                        remark = '1순위' if station_id == first_priority else '2순위'

                                        data_for_all.append({
                                            'time': tm,
                                            'DMS_CODE': dms_code,
                                            '측정소명': dms_name, # dms_name
                                            'stdid': station_id,
                                            'temp': temp,
                                            'wdspeed': wdspeed,
                                            'rain_1hr': rain_1hr,
                                            'humidity': humidity,
                                            'remark': remark
                                        })
                else:
                    print(f"Error {response.status_code} occurred. Waiting 1 hour before retrying...")
                    time.sleep(300)  # 30분 대기
                    continue  # 현재 시간에 대해 다시 시도
                    
            except Exception as e:
                print(f"An error occurred: {e}. Retrying after 1 hour...")
                time.sleep(300)  # 30분 대기
                continue  # 현재 시간에 대해 다시 시도

            time.sleep(1)                        ######## 딜레이

        current_time += timedelta(hours=1)

    return pd.DataFrame(data_for_all)

def adjust_date_for_nighttime(data):
    """19:00~05:00 데이터를 다음 날로 조정."""
    data['date'] = pd.to_datetime(data['time'], format='%Y%m%d%H%M')  # 'time'을 datetime 형식으로 변환
    data['hour'] = data['date'].dt.hour  # 'date'가 datetime 형식이므로 .dt 사용 가능
    data['측정일'] = data['date'].dt.date  # 'adjusted_date'를 날짜만으로 추출
    data.loc[data['hour'] >= 19, '측정일'] += timedelta(days=1)  # 19시 이후는 다음 날로 변경

    # adjusted_date를 datetime 형식으로 변환
    data['측정일'] = pd.to_datetime(data['측정일'])

    return data

def aggregate_daily_data(data):
    # "1순위와 2순위 데이터를 일별로 집계."
    aggregated = (
        data.groupby(['측정일', 'DMS_CODE', '측정소명', 'remark'])
        .agg(
            평균_온도=('temp', 'mean'),
            최고_온도=('temp', 'max'),
            최저_온도=('temp', 'min'),
            강수량_합계=('rain_1hr', 'sum'),
            평균_습도=('humidity', 'mean'),
            평균_풍속=('wdspeed', 'mean')
        )
        .reset_index()
    )
    return aggregated

def calculate_monthly_and_yearly_averages(data):
    
    # 월평균 온도 계산
    data['month'] = data['측정일'].dt.to_period('M')  # 'YYYY-MM' 형식으로 월 추출
    월평균_기온 = data.groupby(['month', 'DMS_CODE'])['평균_온도'].mean().reset_index()
    월평균_기온.rename(columns={'평균_온도': '월평균_기온'}, inplace=True)
    
    # 연평균 온도 계산
    data['year'] = data['측정일'].dt.year  # 연도 추출
    연평균_기온 = data.groupby(['year', 'DMS_CODE'])['평균_온도'].mean().reset_index()
    연평균_기온.rename(columns={'평균_온도': '연평균_기온'}, inplace=True)
    
    # 결과 병합
    data = pd.merge(data, 월평균_기온, on=['month', 'DMS_CODE'], how='left')
    data = pd.merge(data, 연평균_기온, on=['year', 'DMS_CODE'], how='left')
    
    return data

def process_and_fill_gaps(aggregated_data):
    """1순위 데이터를 우선 사용하고, 결측치를 2순위, 3순위, 4순위로 보완."""    
    result = []
    
    for dms_code in desired_dms_codes:
        # 1순위 및 2순위 데이터 선택
        first_priority = aggregated_data[
            (aggregated_data['DMS_CODE'] == dms_code) & (aggregated_data['remark'] == '1순위')
        ].set_index('측정일')
        
        second_priority = aggregated_data[ 
            (aggregated_data['DMS_CODE'] == dms_code) & (aggregated_data['remark'] == '2순위')
        ].set_index('측정일')

        # 1순위 데이터를 기준으로 초기화
        merged = first_priority.copy()

        # 결측치가 있는 항목을 2순위 데이터로 보완
        for col in ['평균_온도', '최고_온도', '최저_온도', '강수량_합계', '평균_습도', '평균_풍속']:
            if col in merged.columns:
                # 2순위 보완
                if merged[col].isna().any():
                    before_fill = merged[col].isna().sum()
                    if col in second_priority.columns:  # Check if the column exists in second_priority
                        merged[col] = merged[col].combine_first(second_priority[col])
                        after_fill = merged[col].isna().sum()
                        print(f"{dms_code}: {col} - 2순위 보완: 결측치 {before_fill}개 중 {before_fill - after_fill}개 보완 완료.")
                        merged.loc[merged[col].notna() & first_priority[col].isna(), 'remark'] = f'2순위 보완-{col}'


                # 3순위 보완 - 다른 년도의 같은 날짜 1순위 평균값으로 채우기
                if merged[col].isna().any():
                    missing_dates = merged[merged[col].isna()].index
                    for date in missing_dates:
                        # 다른 년도의 같은 날짜 값들의 평균을 계산 (1순위 기준)
                        same_day_data = merged[
                            (merged.index.month == date.month) &
                            (merged.index.day == date.day) &
                            (merged['DMS_CODE'] == dms_code)  # dms_code 기준으로 필터링
                        ]
                        # 평균값 계산
                        avg_value = same_day_data[col].dropna().mean()  # NaN 제외 후 평균 계산
                           
                        if not np.isnan(avg_value):
                            merged.at[date, col] = avg_value
                            merged.at[date, 'remark'] = f'3순위 보완-{col}'
                            print(f"{dms_code}: {col} - 3순위 보완: 날짜 {date}에서 값 {avg_value}로 보완 완료.")
                        else:
                            # 3순위 보완 실패시 바로 4순위로 넘어감
                            print(f"{dms_code}: {col} - 3순위 보완 실패: 날짜 {date}에서 3순위 보완을 진행할 수 없음. 4순위로 진행.")

                # 4순위 보완 - 선형 보간법으로 결측치 보완
                if merged[col].isna().any():
                    before_fill = merged[col].isna().sum()
                    merged[col] = merged[col].interpolate(method='linear')
                    after_fill = merged[col].isna().sum()
                    print(f"{dms_code}: {col} - 4순위 보완 (선형 보간법): 결측치 {before_fill}개 중 {before_fill - after_fill}개 보완 완료.")
                    merged.loc[merged[col].notna() & first_priority[col].isna(), 'remark'] = f'4순위 보완-{col}'

        result.append(merged.reset_index())

    return pd.concat(result, ignore_index=True)

def fetch_and_process_all_years():
    all_raw_data = []  # 모든 년도 데이터를 저장할 리스트
    
    years = [2015]  # , 2016, 2017, 2018, 2019, 2020, 2021, 2022
    for year in years:
        start_date = datetime(year, 3, 31, 19, 0)
        end_date = datetime(year, 4, 1, 5, 0) # (year, 10, 31, 5, 0)

        raw_data = fetch_data(start_date, end_date)
        all_raw_data.append(raw_data)  # 각 년도의 데이터를 리스트에 추가

    # 모든 년도의 데이터를 하나로 합친 후
    combined_data = pd.concat(all_raw_data, ignore_index=True)

    # 날짜 처리: 각 년도의 데이터가 합쳐졌으므로 날짜 조정
    adjusted_data = adjust_date_for_nighttime(combined_data)
    
    # 일별 집계
    aggregated_data = aggregate_daily_data(adjusted_data)

    # 결측치 처리: 1순위, 2순위, 3순위, 4순위 순으로 처리
    filled_data = process_and_fill_gaps(aggregated_data)

    # 최종 데이터 처리
    final_data = calculate_monthly_and_yearly_averages(filled_data)

    return final_data


if __name__ == "__main__":
    final_processed_data = fetch_and_process_all_years()

    # 'month, year' 컬럼을 제외하고 저장
    final_processed_data.drop(columns=['month', 'year'], inplace=True, errors='ignore')

    # 열 이름 설정 및 순서 재정렬
    colname = ['측정일', 'DMS_CODE', '측정소명', '비고', '평균_온도', '최고_온도', '최저_온도', '강수량_합계', '평균_습도', '평균_풍속', '월평균_기온', '연평균_기온']
    final_processed_data.columns = colname
    final_processed_data = final_processed_data[['측정일', 'DMS_CODE','측정소명', '평균_온도', '최고_온도', '최저_온도', '강수량_합계', '평균_습도', '평균_풍속', '월평균_기온', '연평균_기온', '비고']]

    # 결과 저장
    final_processed_data.to_excel("F:\박정현\ML\Mosquito\sample6_mosquito.xlsx", index=False)
    print("All data processing complete. Saved to 'sample6_mosquito.xlsx'.")
