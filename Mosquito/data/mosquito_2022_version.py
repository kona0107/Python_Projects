import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# 엑셀 파일 읽기
file_path = r'/home/seunjung1216/매개모기/DMS_관측소_매핑.xlsx'
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
        'dms_name': dms_name
    }

# 원하는 DMS CODE 목록
desired_dms_codes = df['DMS CODE'].tolist()  # 모든 DMS CODE를 목록에 추가

# API 기본 URL
domain = "https://apihub.kma.go.kr/api/typ01/url/awsh.php?"
var = 'var='
option = "authKey=_2OLXOOCTRiji1zjgs0Y-w"  # 인증키

def fetch_data(start_date, end_date):
    """데이터를 한 번에 가져오는 함수."""
    data_for_all = []
    current_time = start_date

    while current_time <= end_date:
        # 19시부터 5시까지만 데이터 접근
        if current_time.hour >= 19 or current_time.hour <= 5:
            tm = current_time.strftime("%Y%m%d%H%M")
            url = f"{domain}{var}&tm={tm}&stn=0&{option}"

            print(f"Fetching data for {current_time.strftime('%Y-%m-%d %H:%M')} with URL: {url}")

            try:
                response = requests.get(url)
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
                                            'DMS_CODE': dms_code,  # DMS CODE 추가,
                                            '측정소명': dms_name,
                                            'stdid': station_id,   # 'stdid' 열 추가
                                            'temp': temp,
                                            'wdspeed': wdspeed,
                                            'rain_1hr': rain_1hr,
                                            'humidity': humidity,
                                            'remark': remark  # remark 열 추가
                                        })
                else:
                    print(f"Error {response.status_code} occurred. Waiting 1 hour before retrying...")
                    time.sleep(300)  # 30분 대기
                    continue  # 현재 시간에 대해 다시 시도

            except Exception as e:
                print(f"An error occurred: {e}. Retrying after 1 hour...")
                time.sleep(300)  # 30분 대기
                continue  # 현재 시간에 대해 다시 시도

            time.sleep(1)  # API 호출 간의 딜레이

        current_time += timedelta(hours=1)

    # 리스트를 DataFrame으로 변환
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
    """1순위와 2순위 데이터를 일별로 집계."""
    aggregated = (
        data.groupby(['측정일', 'DMS_CODE', '측정소명', 'remark'])
        .agg(
            TMP=('temp', 'mean'),
            TMAX=('temp', 'max'),
            TMIN=('temp', 'min'),
            PCP=('rain_1hr', 'sum'),
            HUM=('humidity', 'mean'),
            WS=('wdspeed', 'mean')
        )
        .reset_index()
    )
    return aggregated


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
        for col in ['TMP', 'TMAX', 'TMIN', 'PCP', 'HUM', 'WS']:
            if col in merged.columns:
                # 보완 전 결측치 위치 추적
                missing_before_fill = merged[col].isna()

                before_fill = missing_before_fill.sum()
                merged[col] = merged[col].combine_first(second_priority[col])

                # 보완 후 채워진 값의 위치
                filled_indices = missing_before_fill & merged[col].notna()

                after_fill = merged[col].isna().sum()
                print(f"{dms_code}: {col} - 2순위 보완: 결측치 {before_fill}개 중 {before_fill - after_fill}개 보완 완료.")

                # 'remark' 열에 2순위 보완으로 덮어쓰기 (보완된 값들에 한정)
                merged.loc[filled_indices, 'remark'] = f'2순위 보완-{col}'

                # 2순위로도 결측치가 남아있다면, 같은 날짜 다른 연도의 데이터를 사용하여 보완 (3순위)
                missing_dates = merged[merged[col].isna()].index
                for date in missing_dates:
                    # 3순위 보완이 아직 처리되지 않은 날짜에 대해서만 진행
                    if '3순위 보완' not in str(merged.at[date, 'remark']):
                        # 다른 연도의 같은 날짜 값들의 평균을 계산 (1순위 및 2순위와 무관)
                        same_day_data = merged[
                            (merged.index.month == date.month) &
                            (merged.index.day == date.day) &
                            (merged['DMS_CODE'] == dms_code)  # dms_code 기준으로 필터링
                        ]

                        # 평균값 계산
                        avg_value = same_day_data[col].dropna().mean()  # NaN 제외 후 평균 계산

                        # avg_value가 NaN인 경우 3순위 보완을 진행하지 않음
                        if not np.isnan(avg_value):
                            # 결측치가 있을 경우에만 채우기
                            if pd.isna(merged.at[date, col]):
                                merged.at[date, col] = avg_value
                                merged.at[date, 'remark'] = f'3순위 보완-{col}'  # 3순위 보완으로 표시
                                print(f"{dms_code}: {col} - 3순위 보완: 날짜 {date}에서 값 {avg_value}로 보완 완료.")
                            
                            else:
                                merged.at[date, 'remark'] += f'3순위 보완-{col}'  # 3순위 보완으로 표시
                                print(f"{dms_code}: {col} - 3순위 보완: 날짜 {date}에서 값 {avg_value}로 보완 완료.")
                        else:
                            # 3순위 보완 실패시 바로 4순위로 넘어감
                            print(f"{dms_code}: {col} - 3순위 보완 실패: 날짜 {date}에서 3순위 보완을 진행할 수 없음. 4순위로 진행.")

                            # 4순위 보완 (선형 보간법) 진행
                            missing_dates_4th = merged[merged[col].isna()].index
                            if not missing_dates_4th.empty:
                                before_fill_4th = merged[col].isna().sum()
                                merged[col] = merged[col].interpolate(method='linear')
                                after_fill_4th = merged[col].isna().sum()
                                print(f"{dms_code}: {col} - 4순위 보완 (선형 보간법): 결측치 {before_fill_4th}개 중 {before_fill_4th - after_fill_4th}개 보완 완료.")

                                # 'remark' 열에 '4순위'를 추가
                                merged.loc[merged[col].notna(), 'remark'] = merged.loc[merged[col].notna(), 'remark'].apply(
                                    lambda x: f'{x}, 4순위 보완-{col}' if x else f'4순위 보완-{col}')


        # 결과 반환 시 remark를 그대로 사용
        result.append(merged.reset_index())

    return pd.concat(result, ignore_index=True)

def calculate_monthly_and_yearly_averages(data):
    """월평균과 연평균 온도를 계산하는 함수."""
    # 월평균 온도 계산
    data['month'] = data['측정일'].dt.to_period('M')  # 'YYYY-MM' 형식으로 월 추출
    monthly_avg = data.groupby(['month', 'DMS_CODE'])['TMP'].mean().reset_index()
    monthly_avg.rename(columns={'TMP': 'Temp_1_monthly_mean'}, inplace=True)

    # 연평균 온도 계산
    data['year'] = data['측정일'].dt.year  # 연도 추출
    yearly_avg = data.groupby(['year', 'DMS_CODE'])['TMP'].mean().reset_index()
    yearly_avg.rename(columns={'TMP': 'Temp_1_yearly_mean'}, inplace=True)

    # 결과 병합
    data = pd.merge(data, monthly_avg, on=['month', 'DMS_CODE'], how='left')
    data = pd.merge(data, yearly_avg, on=['year', 'DMS_CODE'], how='left')

    return data


def fetch_and_process_all_years():
    all_raw_data = []  # 모든 년도 데이터를 저장할 리스트

    # 2015년부터 2022년까지 모든 데이터를 가져오기
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    for year in years:
        start_date = datetime(year, 3, 31, 19, 0)
        end_date = datetime(year, 4, 1, 5, 0)

        raw_data = fetch_data(start_date, end_date)
        all_raw_data.append(raw_data)  # 각 년도의 데이터를 리스트에 추가

    # 모든 년도의 데이터를 하나로 합친 후
    combined_data = pd.concat(all_raw_data, ignore_index=True)
    
    # combined_data를 엑셀 파일로 저장
    combined_data.to_csv('2015_raw_data.csv', index=False, encoding = 'cp949')

    # 날짜 처리: 각 년도의 데이터가 합쳐졌으므로 날짜 조정
    adjusted_data = adjust_date_for_nighttime(combined_data)

    # 일별 집계
    aggregated_data = aggregate_daily_data(adjusted_data)
    
    aggregated_data.to_excel("2015_daily_data.xlsx", index=False)

    # 결측치 처리: 1순위, 2순위, 3순위, 4순위 순으로 처리
    filled_data = process_and_fill_gaps(aggregated_data)

    # 최종 데이터 처리
    final_data = calculate_monthly_and_yearly_averages(filled_data)

    return final_data


# 모든 데이터를 한 번에 처리
if __name__ == "__main__":
    final_processed_data = fetch_and_process_all_years()

    # 'month, year' 컬럼을 제외하고 저장
    final_processed_data.drop(columns=['month', 'year'], inplace=True, errors='ignore')

    # 열 이름 설정 및 순서 재정렬
    colname = ['측정일', 'DMS_CODE', '측정소명', 'remark', 'TMP', 'TMAX', 'TMIN', 'PCP', 'HUM', 'WS', 'Temp_1_monthly_mean', 'Temp_1_yearly_mean']
    final_processed_data.columns = colname
    final_processed_data = final_processed_data[['측정일', 'DMS_CODE','측정소명', 'TMP', 'TMAX', 'TMIN', 'PCP', 'HUM', 'WS', 'Temp_1_monthly_mean', 'Temp_1_yearly_mean', 'remark']]


    # 결과 저장
    final_processed_data.to_excel("2015.xlsx", index=False)
    print("All data processing complete. Saved to '2015_2022_mosquito.xlsx'.")