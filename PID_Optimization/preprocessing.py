import pandas as pd
import numpy as np
import glob
import os
from joblib import Parallel, delayed

# --- 1. 상수 정의 (사용자 설정) ---

# !!! (필수) 테스트할 시간 단위를 선택하세요:
RESAMPLE_FREQ = '1T'   # '1T'(1분), '5T'(5분), '30S'(30초), '1S'(1초)
# (1초 방식일 때만 사용됨)
SMOOTHING_WINDOW_SEC_1S = 60 # 1초 데이터용 이동 평균 창 (초)

# (필수) 다운로드할 날짜 범위
FILTER_START_DATE = '2025-09-11' # 9월 11일
FILTER_END_DATE = '2025-11-11'   # 11/11까지 포함

# (필수) 사용할 CPU 코어 수 (n_jobs)
try:
    N_JOBS = os.cpu_count() - 2 if os.cpu_count() > 1 else 1
except:
    N_JOBS = 3 # 기본값

# 화성/ 평택 / 기흥 선택
SITE = 'HS'
# SITE = 'PT'

# !!! (필수) 기타 컬럼명
TIME_COL = 'act_time'
LEVEL_COL_A = 'Level_A'
LEVEL_COL_B = 'Level_B'

if SITE == 'PT':
    V_TANK_A = 7154  # 화성 A 집수조 총 용량 (m³)
    V_TANK_B = 2944  
    OUTFLOW_COLS = [
    '무기1기', '무기2기', '무기3기', '무기4기', '무기5기',
    '무기6기', '무기7기', '무기8기', '무기9기', '무기10기'
    ]# 화성 B 집수조 총 용량 (m³)
elif SITE == 'HS':
    V_TANK_A = 3520  # 평택 A 집수조 총 용량 (m³)
    V_TANK_B = 3455  # 평택 B 집수조 총 용량 (m³)
    OUTFLOW_COLS = [
    '유기1기', '유기2기', '유기3기', '유기4기', '유기5기',
    '유기6기', '유기7기', '유기8기', '유기9기', '유기10기'
]
    
# 경로 설정
BASE_FOLDER = "/config/work/개인폴더/박정현/PID_control"
DATA_FOLDER = os.path.join(BASE_FOLDER, "RAW_DATA", SITE)
OUTPUT_DATA_FOLDER = os.path.join(BASE_FOLDER, "PROCESSED_DATA", SITE)
OUTPUT_FILENAME_PREFIX = f"processed_pid_dataset_{RESAMPLE_FREQ}_{SITE}_CALC_IQR" # 파일명 변경


# --- 2. 함수 정의: 데이터 로드 (1초 중복 제거 포함) ---
def load_and_merge_data(folder_path, file_pattern="*.csv", date_strings_to_load=None):
    """(수정) 파일명 리스트(date_strings_to_load)에 해당하는 '모든' 파일을
    찾아서 로드하고 병합합니다. (예: _0911_ 3개 파일 모두)
    """
    all_files = glob.glob(os.path.join(folder_path, file_pattern))
    if not all_files:
        return pd.DataFrame() 

    files_to_load = []
    if date_strings_to_load:
        for f_path in all_files:
            filename = os.path.basename(f_path)
            if any(date_str in filename for date_str in date_strings_to_load):
                files_to_load.append(f_path)
        if not files_to_load:
            return pd.DataFrame()
    else:
        files_to_load = all_files

    df_list = []
    for file in files_to_load: 
        try:
            df_part = pd.read_csv(file)
            df_list.append(df_part)
        except Exception as e:
            print(f"  파일 로드 실패: {file}, 오류: {e}")
            
    if not df_list: return pd.DataFrame()

    merged_df = pd.concat(df_list, ignore_index=True)
    
    if TIME_COL not in merged_df.columns:
        print(f"  오류: 시간 컬럼 '{TIME_COL}'을 찾을 수 없습니다.")
        return pd.DataFrame()

    merged_df[TIME_COL] = pd.to_datetime(merged_df[TIME_COL])
    
    original_rows = len(merged_df)
    merged_df[TIME_COL] = merged_df[TIME_COL].dt.floor('S')
    merged_df = merged_df.drop_duplicates(subset=[TIME_COL], keep='first')
    new_rows = len(merged_df)
    if original_rows > new_rows:
        print(f"  [처리] {original_rows - new_rows}행의 중복 데이터(1초 단위)를 제거했습니다.")
    
    merged_df = merged_df.sort_values(by=TIME_COL).reset_index(drop=True)
    return merged_df

# --- 3. '하루치' 작업을 수행하는 워커(Worker) 함수 ---
def process_day_task(day_date, freq, smoothing_window):
    """
    (Worker) 하루치 1초 데이터를 로드, 필터링, 처리(Resample/Rolling)
    (수정) diff(), IQR, 역산 로직을 '제거'하고 재집계만 수행
    """
    
    print(f"--- [Core] {day_date:%Y-%m-%d}일 데이터 로드/재집계 시작 ---")
    
    # 1. 하루치 파일 로드 (예: _0911_ 패턴 3개)
    date_string_to_load = [day_date.strftime('_%m%d_')]
    raw_df_day = load_and_merge_data(
        DATA_FOLDER, 
        file_pattern="*.csv", 
        date_strings_to_load=date_string_to_load
    )
    
    if raw_df_day.empty:
        print(f"  정보: {day_date:%Y-%m-%d}일 데이터를 찾지 못했습니다. 건너뜁니다.")
        return pd.DataFrame()
        
    # 2. 1초 데이터를 'act_time' 기준으로 정확히 24시간 필터링 (겹침 제거)
    day_start_str = day_date.strftime('%Y-%m-%d 00:00:00')
    day_end_str = (day_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')
    
    raw_df_day = raw_df_day[
        (raw_df_day[TIME_COL] >= day_start_str) &
        (raw_df_day[TIME_COL] < day_end_str)
    ].copy()

    if raw_df_day.empty:
        print(f"  정보: {day_date:%Y-%m-%d}일 범위 내 데이터가 없습니다.")
        return pd.DataFrame()

    # 3. 1초 -> N초/N분 처리 (재집계 또는 이동평균)
    try:
        valid_outflow_cols = [col for col in OUTFLOW_COLS if col in raw_df_day.columns]
        raw_df_day['Q_out_total_hr'] = raw_df_day[valid_outflow_cols].sum(axis=1)
        cols_needed = [TIME_COL, LEVEL_COL_A, LEVEL_COL_B, 'Q_out_total_hr']
        if not all(col in raw_df_day.columns for col in cols_needed):
            print(f"  오류: {day_date:%Y-%m-%d} 필수 컬럼이 없습니다.")
            return pd.DataFrame()
    except Exception as e:
        print(f"  오류: {day_date:%Y-%m-%d} 1초 데이터 전처리 중 실패: {e}")
        return pd.DataFrame()
    
    # --- (핵심 수정) 로직 분기 ---
    
    raw_df_day = raw_df_day.set_index(TIME_COL)
    
    if freq == '1S':
        # B1. 1초(1S) 로직: 이동 평균 (Smoothing)
        # (주의: 1초는 여전히 diff() 증폭 문제가 있을 수 있음)
        cols_to_process = [LEVEL_COL_A, LEVEL_COL_B, 'Q_out_total_hr']
        df_processed = raw_df_day[cols_to_process].copy()
        
        df_processed[LEVEL_COL_A] = df_processed[LEVEL_COL_A].rolling(window=smoothing_window, min_periods=1, center=True).mean()
        df_processed[LEVEL_COL_B] = df_processed[LEVEL_COL_B].rolling(window=smoothing_window, min_periods=1, center=True).mean()
    else:
        # B2. 1초 초과 (1T, 5T, 30S) 로직: 재집계 (Resampling)
        cols_to_resample = [LEVEL_COL_A, LEVEL_COL_B, 'Q_out_total_hr']
        df_processed = raw_df_day[cols_to_resample].resample(freq).mean()
        df_processed = df_processed.dropna()
        if df_processed.empty:
            return pd.DataFrame()
        
    # C. (수정) diff(), 역산, IQR 계산을 '제거'하고 재집계된 데이터만 반환
    return df_processed

# --- 4. (실행) 1단계 스크립트 실행 (joblib 수정) ---
if __name__ == "__main__":
    
    print(f"--- 1단계 (레벨 역산 + IQR 방식, {RESAMPLE_FREQ} 단위): 데이터 전처리 시작 ---")

    # 1. '일' 단위로 날짜 목록 생성
    try:
        daily_task_list = pd.date_range(start=FILTER_START_DATE, end=FILTER_END_DATE, freq='D')
    except Exception as e:
        print(f"오류: 날짜 범위 설정 실패 ({FILTER_START_DATE}~{FILTER_END_DATE}). {e}")
        exit()
        
    print(f"총 {len(daily_task_list)}일의 데이터를 '일' 단위로 {N_JOBS}개 코어를 사용하여 병렬 처리합니다...")

    # 2. (루프 A) '일'별로 병렬 처리 (joblib)
    parallel_executor = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)
    
    all_processed_dfs = parallel_executor(
        delayed(process_day_task)(
            day_date, 
            freq=RESAMPLE_FREQ, 
            smoothing_window=SMOOTHING_WINDOW_SEC_1S
        )
        for day_date in daily_task_list
    )
        
    print("\n[1단계] 모든 병렬 처리 완료. 최종 병합 및 계산 시작...")

    # 3. (병합) 모든 일별 처리 데이터를 하나로 병합
    if not all_processed_dfs:
        print("오류: 처리할 데이터가 없습니다. 스크립트를 종료합니다.")
        exit()
        
    final_df = pd.concat(all_processed_dfs)
    final_df = final_df.sort_index() 
    
    # 4. '비상 대응' 기간 등 상세 필터링 (필요시)
    # (현재는 비활성화)

    # 5. (수정) '정상' 데이터에 대해서만 역산 및 IQR 수행
    print("[1단계] '전체 2개월' 데이터에 대해 `diff`, `역산`, `IQR` 일괄 계산 시작...")
    
    # C. 유입량 역산
    final_df['Volume_A_m3'] = (final_df[LEVEL_COL_A] / 100.0) * V_TANK_A
    final_df['Volume_B_m3'] = (final_df[LEVEL_COL_B] / 100.0) * V_TANK_B
    final_df['Volume_total_m3'] = final_df['Volume_A_m3'] + final_df['Volume_B_m3']
    
    # (버그 수정) diff()가 여기서 '일괄적'으로 한번만 실행됨
    final_df['Volume_delta_m3'] = final_df['Volume_total_m3'].diff()

    if RESAMPLE_FREQ == '1S':
        conversion_factor = 3600.0
    else:
        resample_minutes = pd.to_timedelta(RESAMPLE_FREQ).total_seconds() / 60.0
        conversion_factor = 60.0 / resample_minutes
    
    final_df['Q_in_calc_hr'] = (final_df['Volume_delta_m3'] * conversion_factor) + final_df['Q_out_total_hr']

    # D. IQR 방식 이상치 처리 (일괄 처리)
    Q1 = final_df['Q_in_calc_hr'].quantile(0.25)
    Q3 = final_df['Q_in_calc_hr'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    print(f"  [처리] IQR 경계 ({RESAMPLE_FREQ}): {lower_bound:.2f} ~ {upper_bound:.2f}")

    outliers_low_count = (final_df['Q_in_calc_hr'] < lower_bound).sum()
    outliers_high_count = (final_df['Q_in_calc_hr'] > upper_bound).sum()
    total_outliers = outliers_low_count + outliers_high_count
    
    if total_outliers > 0:
        print(f"  [처리] IQR: 총 {total_outliers}개 항목(하한 {outliers_low_count}개, 상한 {outliers_high_count}개)이 경계값으로 대체(Clipping)됩니다.")
    else:
        print("  [처리] IQR: 이상치가 발견되지 않았습니다.")
        
    final_df['Q_in_calc_hr'] = final_df['Q_in_calc_hr'].clip(lower=lower_bound, upper=upper_bound)
    
    # E. 후처리
    final_df = final_df.fillna(0) # (버그 수정) .diff()로 인한 '첫 번째 행'(9/11 00:00) NaN만 0으로 처리
    final_df['Q_in_calc_hr'] = final_df['Q_in_calc_hr'].clip(lower=0)
    
    print(f"'{RESAMPLE_FREQ}' 단위 유입량 역산 및 IQR 처리 완료.")

    # 6. 최종 저장
    print(f"\n[1단계] --- 최종 2개월치 컬럼 통계 ({RESAMPLE_FREQ} 단위, IQR 적용됨) ---")
    print(final_df[['Q_out_total_hr', 'Q_in_calc_hr']].describe())

    print("\n[1단계] 파일 저장 시작...")
    if not os.path.exists(OUTPUT_DATA_FOLDER):
        os.makedirs(OUTPUT_DATA_FOLDER)
    
    COLS_TO_SAVE = [LEVEL_COL_A, LEVEL_COL_B, 'Q_out_total_hr', 'Q_in_calc_hr']
    try:
        start_date = final_df.index[0]
        date_str = start_date.strftime('%Y%m%d') # 0911
        
        output_filename = f"{OUTPUT_FILENAME_PREFIX}_{date_str}_2M.csv" # 2개월치(2M) 표시
        save_path = os.path.join(OUTPUT_DATA_FOLDER, output_filename)
        
        final_df[COLS_TO_SAVE].to_csv(save_path, index=True, index_label=TIME_COL, encoding='utf-8-sig')
        
        print(f"[1단계] 성공: {len(COLS_TO_SAVE)}개 컬럼이 {save_path}에 저장되었습니다.")
    except Exception as e:
        print(f"[1단계] 오류: 파일 저장 실패. {e}")