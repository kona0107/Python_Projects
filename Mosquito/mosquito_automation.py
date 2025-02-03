import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# 실행 시작 시간 기록
total_start_time = time.time()

# 엑셀 파일 읽기
file_path = 'DMS_관측소_매핑.xlsx'
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

# 원하는 DMS CODE 목록 (전체 DMS CODE 사용)
desired_dms_codes = df['DMS CODE'].tolist()

# API 기본 URL
domain = "https://apihub.kma.go.kr/api/typ01/url/awsh.php?"
var = 'var='
option = "authKey=s5DFRYyOQqKQxUWMjlKi9g"  # 인증키 추가 필요

def fetch_all_data(start_date, end_date):
    """전체 데이터를 가져온 후 필요한 DMS CODE와 매핑된 데이터를 필터링."""
    data_for_period = []
    current_time = start_date

    while current_time <= end_date:
        tm = current_time.strftime("%Y%m%d%H%M")
        url = f"{domain}{var}&tm={tm}&stn=0&{option}"

        print(f"Fetching data for {current_time.strftime('%Y-%m-%d %H:%M')} with URL: {url}")
        response = requests.get(url)

        if response.status_code == 200:
            text_data = response.text.strip()
            if text_data:
                lines = text_data.splitlines()
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        fields = line.split()
                        station_id = int(fields[1])
                        temp = np.nan if float(fields[2]) == -99 else float(fields[2])
                        wdspeed = np.nan if float(fields[4]) == -99 else float(fields[4])
                        rain_1hr = np.nan if float(fields[6]) == -99 else float(fields[6])
                        humidity = np.nan if float(fields[7]) == -99 else float(fields[7])

                        # DMS CODE와 매핑된 1순위 및 2순위 관측소 필터링
                        for dms_code in desired_dms_codes:
                            first_priority = dms_station_mapping[dms_code]['기상 1순위']
                            second_priority = dms_station_mapping[dms_code]['기상 2순위']
                            if station_id == first_priority or station_id == second_priority:
                                remark = '1순위' if station_id == first_priority else '2순위'
                                data_for_period.append({
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

        time.sleep(0.2)  # 딜레이
        current_time += timedelta(hours=1)

    return pd.DataFrame(data_for_period)


def adjust_date_for_nighttime(data):
    """19:00~05:00 데이터를 다음 날로 조정."""
    data['date'] = pd.to_datetime(data['time'], format='%Y%m%d%H%M')  # 'time'을 datetime 형식으로 변환
    data['hour'] = data['date'].dt.hour  # 'date'가 datetime 형식이므로 .dt 사용 가능
    data['adjusted_date'] = data['date'].dt.date # 'adjusted_date'를 날짜만으로 추출
    data.loc[data['hour'] >= 19, 'adjusted_date'] += timedelta(days=1)  # 19시 이후는 다음 날로 변경
    data['adjusted_date'] = pd.to_datetime(data['adjusted_date'])
    return data


def aggregate_daily_data(data):
    """일별 데이터 집계."""
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
    """결측치 보완: 1순위 → 2순위 → 다른 연도 평균 → 선형 보간."""
    result = []
    
    for dms_code in desired_dms_codes:
        first_priority = aggregated_data[
            (aggregated_data['dms_code'] == dms_code) & (aggregated_data['remark'] == '1순위')
        ].set_index('adjusted_date')

        second_priority = aggregated_data[
            (aggregated_data['dms_code'] == dms_code) & (aggregated_data['remark'] == '2순위')
        ].set_index('adjusted_date')

        merged = first_priority.copy()

        for col in ['temp_avg', 'temp_max', 'temp_min', 'rain_sum', 'humidity_avg', 'wdspeed_avg']:
            before_fill_2nd = merged[col].isna().sum()
            merged[col] = merged[col].combine_first(second_priority[col])
            after_fill_2nd = merged[col].isna().sum()
            print(f"{dms_code}: {col} - 2순위로 보완된 결측치: {before_fill_2nd - after_fill_2nd}개")
            missing_dates = merged[merged[col].isna()].index
            for date in missing_dates:
                same_day_data = aggregated_data[
                    (aggregated_data['adjusted_date'].dt.month == date.month) &
                    (aggregated_data['adjusted_date'].dt.day == date.day) &
                    (aggregated_data['dms_code'] == dms_code) &
                    (aggregated_data['remark'] == '1순위')
                ]
                avg_value = same_day_data[col].mean()
                if not np.isnan(avg_value):
                    before_fill_3rd = merged[col].isna().sum()
                    merged.at[date, col] = avg_value
                    after_fill_3rd = merged[col].isna().sum()
                    print(f"{dms_code}: {col} - 3순위로 보완된 결측치: {before_fill_3rd - after_fill_3rd}개")

            before_fill_4th = merged[col].isna().sum()
            merged[col] = merged[col].interpolate(method='linear')
            after_fill_4th = merged[col].isna().sum()
            print(f"{dms_code}: {col} - 4순위 보간법으로 보완된 결측치: {before_fill_4th - after_fill_4th}개")


        # 비고 컬럼 생성
        merged['비고'] = merged.apply(update_remark, axis=1)

        result.append(merged.reset_index())

    return pd.concat(result, ignore_index=True)


def main_process(raw_data):
    """전체 처리 파이프라인."""
    raw_data['date'] = pd.to_datetime(raw_data['time'], format='%Y%m%d%H%M')
    adjusted_data = adjust_date_for_nighttime(raw_data)
    aggregated_data = aggregate_daily_data(adjusted_data)
    filled_data = process_and_fill_gaps(aggregated_data)
    return filled_data


if __name__ == "__main__":
    total_start_time = time.time()  # 전체 실행 시작 시간
    all_processed_data = []
    years = [2015]
    # , 2016, 2017, 2018, 2019, 2020, 2021, 2022
    for year in years:
        start_date = datetime(year, 3, 31, 19, 0)
        end_date = datetime(year, 4, 1, 5, 0)

       # 연도별 데이터 가져오기 및 처리
        raw_data = fetch_all_data(start_date, end_date)
        processed_data = main_process(raw_data)

        # remark, month, year 컬럼 제거
        processed_data.drop(columns=['remark', 'month', 'year'], inplace=True, errors='ignore')

        all_processed_data.append(processed_data)

    final_data = pd.concat(all_processed_data, ignore_index=True)
    # 컬럼 이름 설정
    colname = ['날짜', 'DMS_CODE', '평균 온도', '최고 온도', '최저 온도', '강수량 합계', '평균 습도', '평균 풍속', '비고', '월평균 기온', '년평균 기온']
    final_data.columns = colname
    # 열 순서 재정렬
    final_data = final_data[['날짜', 'DMS_CODE', '평균 온도', '최고 온도', '최저 온도', '강수량 합계', '평균 습도', '평균 풍속', '월평균 기온', '년평균 기온', '비고']]
    # 엑셀 파일로 저장
    final_data.to_excel("2015J_mosquito_average.xlsx", index=False)
    print("All data processing complete. Saved to '2015_2022_mosquito_average.xlsx'.")
    # 전체 실행 시간 출력
    total_end_time = time.time()
    elapsed_time = total_end_time - total_start_time
    print(f"전체 실행 시간: {elapsed_time:.2f}초")
