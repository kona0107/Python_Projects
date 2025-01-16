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
    first_priority = row['1순위 관측소 코드']
    second_priority = row['2순위 관측소 코드']
    
    dms_station_mapping[dms_code] = {
        '기상 1순위': first_priority,
        '기상 2순위': second_priority
    }

# 원하는 DMS CODE 목록
desired_dms_codes = df['DMS CODE'].tolist()

# API 기본 URL 및 인증 키 설정
domain = "https://apihub.kma.go.kr/api/typ01/url/awsh.php?"
var = 'var='
api_keys = ["s5DFRYyOQqKQxUWMjlKi9g", "8CMfLj-MS2SjHy4_jAtkQQ"]  # 인증키 리스트
current_key_index = 0  # 현재 사용 중인 API 키 인덱스
api_call_count = 0  # 호출 횟수

def switch_api_key():
    """API 호출 횟수가 12,000번을 초과하면 키를 변경."""
    global current_key_index
    current_key_index = (current_key_index + 1) % len(api_keys)
    print(f"Switching to API key: {api_keys[current_key_index]}")

def fetch_data_for_month(start_date, end_date):
    """한 달 데이터를 한 번에 가져오는 함수."""
    global api_call_count
    data_for_month = []
    current_time = start_date

    while current_time <= end_date:
        # 19시부터 5시까지만 데이터 접근
        if current_time.hour >= 19 or current_time.hour <= 5:
            tm = current_time.strftime("%Y%m%d%H%M")
            url = f"{domain}{var}&tm={tm}&stn=0&authKey={api_keys[current_key_index]}"
            
            print(f"Fetching data for {current_time.strftime('%Y-%m-%d %H:%M')} with URL: {url}")
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
                                if station_id == first_priority or station_id == second_priority:
                                    temp = np.nan if float(fields[2]) == -99 else float(fields[2])
                                    wdspeed = np.nan if float(fields[4]) == -99 else float(fields[4])
                                    rain_1hr = np.nan if float(fields[6]) == -99 else float(fields[6])
                                    humidity = np.nan if float(fields[7]) == -99 else float(fields[7])

                                    remark = '1순위' if station_id == first_priority else '2순위'

                                    data_for_month.append({
                                        'time': tm,
                                        'dms_code': dms_code,
                                        'stdid': station_id,
                                        'temp': temp,
                                        'wdspeed': wdspeed,
                                        'rain_1hr': rain_1hr,
                                        'humidity': humidity,
                                        'remark': remark
                                    })
            else:
                print(f"Failed to fetch data for {current_time.strftime('%Y-%m-%d %H:%M')} (Status: {response.status_code})")
            
            time.sleep(1)                        ######## 딜레이
        current_time += timedelta(hours=1)

    return pd.DataFrame(data_for_month)

def adjust_date_for_nighttime(data):
    """19:00~05:00 데이터를 다음 날로 조정."""
    data['date'] = pd.to_datetime(data['time'], format='%Y%m%d%H%M')
    data['hour'] = data['date'].dt.hour
    data['adjusted_date'] = data['date'].dt.date
    data.loc[data['hour'] >= 19, 'adjusted_date'] += timedelta(days=1)
    data['adjusted_date'] = pd.to_datetime(data['adjusted_date'])
    return data

def aggregate_daily_data(data):
    """1순위와 2순위 데이터를 일별로 집계.""" 
    aggregated = (
        data.groupby(['adjusted_date', 'dms_code', 'remark'])
        .agg(
            temp_avg=('temp', 'mean'),
            temp_max=('temp', 'max'),
            temp_min=('temp', 'min'),
            rain_sum=('rain_1hr', 'sum'),
            humidity_avg=('humidity', 'mean'),
            wdspeed_avg=('wdspeed', 'mean')
        )
        .reset_index()
    )
    return aggregated

def filter_active_period(data):
    """4월~10월 데이터만 필터링."""
    return data[(data['adjusted_date'].dt.month >= 4) & (data['adjusted_date'].dt.month <= 10)]

def calculate_monthly_and_yearly_averages(data):
    """월평균과 연평균 온도를 계산하는 함수."""
    # 월평균 온도 계산
    data['month'] = data['adjusted_date'].dt.to_period('M')  # 'YYYY-MM' 형식으로 월 추출
    monthly_avg = data.groupby(['month', 'dms_code'])['temp_avg'].mean().reset_index()
    monthly_avg.rename(columns={'temp_avg': 'monthly_avg_temp'}, inplace=True)
    
    # 연평균 온도 계산
    data['year'] = data['adjusted_date'].dt.year  # 연도 추출
    yearly_avg = data.groupby(['year', 'dms_code'])['temp_avg'].mean().reset_index()
    yearly_avg.rename(columns={'temp_avg': 'yearly_avg_temp'}, inplace=True)
    
    # 결과 병합
    data = pd.merge(data, monthly_avg, on=['month', 'dms_code'], how='left')
    data = pd.merge(data, yearly_avg, on=['year', 'dms_code'], how='left')
    
    return data

def process_and_fill_gaps(aggregated_data):
    """1순위 데이터를 우선 사용하고, 결측치를 2순위, 3순위, 4순위로 보완."""    
    result = []
    
    for dms_code in desired_dms_codes:
        # 1순위 및 2순위 데이터 선택
        first_priority = aggregated_data[
            (aggregated_data['dms_code'] == dms_code) & (aggregated_data['remark'] == '1순위')
        ].set_index('adjusted_date')
        
        second_priority = aggregated_data[ 
            (aggregated_data['dms_code'] == dms_code) & (aggregated_data['remark'] == '2순위')
        ].set_index('adjusted_date')

        # 1순위 데이터를 기준으로 초기화
        merged = first_priority.copy()

        # 결측치가 있는 항목을 2순위 데이터로 보완
        for col in ['temp_avg', 'temp_max', 'temp_min', 'rain_sum', 'humidity_avg', 'wdspeed_avg']:
            if col in merged.columns:
                before_fill = merged[col].isna().sum()
                merged[col] = merged[col].combine_first(second_priority[col])
                after_fill = merged[col].isna().sum()
                print(f"{dms_code}: {col} - 결측치 {before_fill}개 중 {before_fill - after_fill}개 보완 완료.")
                merged.loc[merged[col].notna() & first_priority[col].isna(), 'remark'] = f'2순위 보완-{col}'

        # 3순위 보완 - 다른 년도의 같은 날짜 1순위 평균값으로 채우기
        for col in ['temp_avg', 'temp_max', 'temp_min', 'rain_sum', 'humidity_avg', 'wdspeed_avg']:
            if col in merged.columns:
                missing_dates = merged[merged[col].isna()].index
                for date in missing_dates:
                    # 다른 년도의 같은 날짜 값들의 평균을 계산 (1순위 기준)
                    same_day_data = aggregated_data[
                        (aggregated_data['adjusted_date'].dt.month == date.month) &
                        (aggregated_data['adjusted_date'].dt.day == date.day) &
                        (aggregated_data['dms_code'] == dms_code) &
                        (aggregated_data['remark'] == '1순위')
                    ]
                    avg_value = same_day_data[col].mean()
                    if not np.isnan(avg_value):
                        merged.at[date, col] = avg_value
                        merged.at[date, 'remark'] = f'3순위 보완-{col}'

        # 4순위 보완 - 선형 보간법으로 결측치 보완
        for col in ['temp_avg', 'temp_max', 'temp_min', 'rain_sum', 'humidity_avg', 'wdspeed_avg']:
            if col in merged.columns:
                before_fill = merged[col].isna().sum()
                merged[col] = merged[col].interpolate(method='linear')
                after_fill = merged[col].isna().sum()
                print(f"{dms_code}: {col} - 4순위 보완 (선형 보간법): 결측치 {before_fill}개 중 {before_fill - after_fill}개 보완 완료.")
                merged.loc[merged[col].notna() & first_priority[col].isna(), 'remark'] = f'4순위 보완-{col}'

        result.append(merged.reset_index())

    return pd.concat(result, ignore_index=True)

def main_process(raw_data):
    """전체 처리 파이프라인."""
    raw_data['date'] = pd.to_datetime(raw_data['time'], format='%Y%m%d%H%M')
    
    # 1. 시간 조정
    adjusted_data = adjust_date_for_nighttime(raw_data)
    # 2. 일별 집계
    aggregated_data = aggregate_daily_data(adjusted_data)
    # 3. 결측치 보완
    filled_data = process_and_fill_gaps(aggregated_data)
    # 4. 4월~10월 데이터 필터링
    filtered_data = filter_active_period(filled_data)
    # 5. 월/연평균 계산
    final_data = calculate_monthly_and_yearly_averages(filtered_data)
    return final_data


if __name__ == "__main__":
    all_processed_data = []
    years = [2015]
 # , 2016, 2017, 2018, 2019, 2020, 2021, 2022
    for year in years:
        start_date = datetime(year, 3, 31, 19, 0)
        end_date = datetime(year, 4, 1, 5, 0)

        raw_data = fetch_data_for_month(start_date, end_date)
        processed_data = main_process(raw_data)

        # remark 컬럼을 제외하고 저장
        processed_data.drop(columns=['month', 'year'], inplace=True, errors='ignore')
    
        all_processed_data.append(processed_data)  # 모든 데이터를 리스트에 추가

    # 모든 년도 데이터를 하나로 합친 후 저장
    final_data = pd.concat(all_processed_data, ignore_index=True)
    colname = ['날짜', 'DMS_CODE', '비고', '평균 온도', '최고 온도','최저 온도', '강수량 합계', '평균 습도',  '월평균 기온', '연평균 기온', '평균 풍속']
    final_data.columns = colname
    # 열 순서 재정렬
    final_data = final_data[['날짜', 'DMS_CODE', '평균 온도', '최고 온도', '최저 온도', '강수량 합계', '평균 습도', '평균 풍속', '월평균 기온', '연평균 기온', '비고']]
    final_data.to_excel("F:\박정현\ML\Mosquito\sampel0401_mosquito_average.xlsx", index=False)
    print("All data processing complete. Saved to 'processed_weather_data_2022_years_with_averages.xlsx'.")
