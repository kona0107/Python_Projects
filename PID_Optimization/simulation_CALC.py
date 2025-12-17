import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import glob
import os
# import itertools
# from joblib import Parallel, delayed

# --- 1. 전역 상수 (시뮬레이션 조건 및 경로) ---

# (필수) 시뮬레이션 조건 (45-75% 제약)
L_SP = 60.0
L_MIN = 45.0
L_MAX = 75.0
Q_MIN = 0.0

# (필수) 1단계에서 처리된 데이터의 특성
RESAMPLE_FREQ = '1T' # '1T' (1분) 데이터

# --- (핵심 수정) Kp (비례 이득)와 Ti (적분 시간, 단위: 분) 탐색 ---
# (Kp=9.2, Ki=0.02 -> Ti = Kp*60/Ki = 27,600)
Kp_range = np.linspace(5.0, 15.0, 10)    # Kp (예: 5~15)
Ti_range = np.linspace(10000.0, 50000.0, 10) # Ti (분), 1만 ~ 5만 (27,600 포함)

Q_OUT_COL = 'Q_out_total_hr'
Q_IN_COL = 'Q_in_calc_hr' # '역산' 데이터 사용
TIME_COL_NAME = 'act_time' 
# ---- PT--------
SITE = 'PT'
LEVEL_COL_A = 'L_A_pct'
LEVEL_COL_B = 'L_B_pct'
Q_CAPA = 2100.0
Q_MAX = Q_CAPA * 1
V_TANK_A = 7154
V_TANK_B = 2944

# # ----- HS--------
# SITE = 'HS'
# LEVEL_COL_A = 'L_A_pct'
# LEVEL_COL_B = 'L_B_pct'
# Q_CAPA = 2000.0
# Q_MAX = Q_CAPA * 1
# V_TANK_A = 3520
# V_TANK_B = 3455
# #------------------
V_TOTAL = V_TANK_A + V_TANK_B

# 1단계 파일 접두사 (역산 방식 파일)
FILENAME_PREFIX = f"processed_pid_dataset_{RESAMPLE_FREQ}_CALC_IQR" 
# 경로
BASE_FOLDER = "/config/work/개인폴더/박정현/PID_control"
INPUT_FOLDER = os.path.join(BASE_FOLDER, "PROCESSED_DATA", SITE)
PLOT_OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "PLOTS")

# ======================================================================
# --- 2. (최종 수정) PI 시뮬레이션 함수 (Anti-Windup 적용) ---
# ======================================================================

def run_simulation(Kp, Ti, df_input, initial_level_pct, Q_min, Q_max, L_SP, L_MIN, L_MAX, DELTA_T_MINUTES):
    """
    Kp*(e + 1/Ti * ∫e dt) 종속형 공식 + 복리 누적 + Anti-Windup 적용
    """
    
    q_in_series = df_input[Q_IN_COL].values
    num_steps = len(q_in_series)
    
    level_pct_sim = np.zeros(num_steps)
    q_out_sim = np.zeros(num_steps)
    
    # -------------------------------------------------------
    # 1. 초기화 및 범프리스 전환 (Bumpless Transfer)
    # -------------------------------------------------------
    level_pct_sim[0] = initial_level_pct
    
    # t=0 시점의 오차 및 P항
    initial_error = initial_level_pct - L_SP # (정동작) PV - SP
    initial_p_term = Kp * initial_error
    
    # t=0 시점의 실제 유출량
    initial_q_out = df_input[Q_OUT_COL].iloc[0]
    
    # I항 초기값 역산 (I = MV - P)
    # 이렇게 해야 t=1 시작 시 MV가 튀지 않고 부드럽게 이어짐
    i_term = initial_q_out - initial_p_term    
    
    # t=0 유출량 설정
    q_out_sim[0] = initial_q_out

    # -------------------------------------------------------
    # 2. 시뮬레이션 루프
    # -------------------------------------------------------
    for t in range(1, num_steps):
        # A. 현재 상태 확인
        current_level_pct = level_pct_sim[t-1]
        error = current_level_pct - L_SP # (정동작)
        
        # B. P항 계산
        p_term = Kp * error
        
        # C. I항 '임시' 계산 (아직 확정 아님)
        # 종속형 Ti 공식: (Kp / Ti) * error * dt
        if abs(Ti) > 1e-6:
            i_temp = i_term + (Kp / Ti) * error * DELTA_T_MINUTES
        else:
            i_temp = 0.0 # Ti가 0에 가까우면(오류) 적분 안 함
            
        # D. 가상의 MV 계산
        mv_raw = p_term + i_temp
        
        # E. [핵심] Anti-Windup 및 출력 제한
        if mv_raw > Q_max:
            # 상한 포화: 출력을 Max로 고정하고, I항을 역산하여 묶어둠
            mv = Q_max
            i_term = Q_max - p_term # Back-calculation
            
        elif mv_raw < Q_min:
            # 하한 포화: 출력을 Min으로 고정하고, I항을 역산하여 묶어둠
            mv = Q_min
            i_term = Q_min - p_term # Back-calculation
            
        else:
            # 정상 범위: 계산된 값 그대로 사용하고, I항 누적(갱신)
            mv = mv_raw
            i_term = i_temp
            
        # 최종 결정된 MV 저장
        q_out_sim[t] = mv
        
        # F. 프로세스 모델 계산 (물질 수지)
        # 다음 스텝의 레벨 예측
        current_q_in = q_in_series[t] 
        current_q_out = q_out_sim[t]
        
        # 시간 단위 환산 (분 -> 시)
        DELTA_T_HOUR = DELTA_T_MINUTES / 60.0
        
        current_volume = (current_level_pct / 100.0) * V_TOTAL
        # 유입 - 유출 = 부피 변화량
        volume_delta = (current_q_in - current_q_out) * DELTA_T_HOUR 
        
        next_volume = current_volume + volume_delta
        level_pct_sim[t] = (next_volume / V_TOTAL) * 100.0
        
        # G. 제약 조건 위반 시 즉시 중단 (속도 최적화)
        if not (L_MIN < level_pct_sim[t] < L_MAX):
            return float('inf'), None, None 
# ======================================================================
# --- 3. 메인 실행 함수 ---
# ======================================================================

def main():
    """2단계 시뮬레이션 메인 실행 함수"""
    print("--- 2단계 (Ti 종속형, 시간 단위 통일): 시뮬레이션 시작 ---")
    
    # 1. 1단계에서 생성된 데이터 로드
    try:
        list_of_files = glob.glob(os.path.join(INPUT_FOLDER, f"{FILENAME_PREFIX}*.csv"))
        if not list_of_files:
            raise FileNotFoundError(f"'{INPUT_FOLDER}'에서 '{FILENAME_PREFIX}'로 시작하는 파일을 찾을 수 없습니다.")
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"[2단계] 데이터 로드: {latest_file}")
        df = pd.read_csv(latest_file, index_col=TIME_COL_NAME, parse_dates=True)
        required_cols = [LEVEL_COL_A, LEVEL_COL_B, Q_OUT_COL, Q_IN_COL]
        if not all(col in df.columns for col in required_cols):
            print(f"[2단계] 오류: 로드된 파일에 {required_cols} 컬럼이 모두 존재하지 않습니다.")
            return
    except Exception as e:
        print(f"[2단계] 오류: {latest_file} 파일 로드 실패. {e}")
        return

    # 2. 시뮬레이션 준비 (값 계산)
    try:
        df['Volume_A_m3'] = (df[LEVEL_COL_A] / 100.0) * V_TANK_A
        df['Volume_B_m3'] = (df[LEVEL_COL_B] / 100.0) * V_TANK_B
        df['Volume_total_m3'] = df['Volume_A_m3'] + df['Volume_B_m3']
        df['Level_total_pct'] = (df['Volume_total_m3'] / V_TOTAL) * 100.0
        
        initial_level_pct = df['Level_total_pct'].iloc[0]
        data_min_level = df['Level_total_pct'].min()
        data_max_level = df['Level_total_pct'].max()
        BENCHMARK_VARIANCE = df[Q_OUT_COL].var()
        
        # (수정) DELTA_T_HOUR 계산 (1분 = 1/60 시간)
        delta_t_minutes = (df.index[1] - df.index[0]).total_seconds() / 60.0
        DELTA_T_HOUR = delta_t_minutes / 60.0
        
        print(f"[2단계] 시뮬레이션 준비 완료: Time Step = {delta_t_minutes:.1f} 분 (DELTA_T_HOUR = {DELTA_T_HOUR:.4f})")
        print(f"[2단계] 벤치마크 분산: {BENCHMARK_VARIANCE:.2f}")
        
        if not (initial_level_pct > L_MIN and initial_level_pct < L_MAX):
            print(f" [2단계] 오류: '시작 레벨'({initial_level_pct:.1f}%)이 제약 조건(L_MIN:{L_MIN}, L_MAX:{L_MAX})을 벗어났습니다.")
            return
        
        if not (data_min_level > L_MIN and data_max_level < L_MAX):
             print(f"  (참고: 과거 데이터는 이 제약(Min {data_min_level:.1f}%)을 위반했으나, 시뮬레이션을 시도합니다.)")
            
    except Exception as e:
        print(f"[2D단계] 오류: 시뮬레이션 준비 중 오류 발생. {e}")
        return

    # 3. 그리드 서치 (Grid Search) 실행
    print(f"\n[2단계] 최적 파라미터 탐색 (Grid Search) 시작 (L_MIN={L_MIN}, L_MAX={L_MAX}) ---")
    
    best_params = {'Kp': None, 'Ti': None} 
    min_variance = float('inf')
    best_q_out_sim = None 
    best_level_sim = None 
    
    total_searches = len(Kp_range) * len(Ti_range)
    count = 0

    for Kp in Kp_range:
        for Ti in Ti_range:
            count += 1
            # (수정) DELTA_T_HOUR 전달
            variance, q_out_sim_result, level_sim_result = run_simulation(
                Kp, Ti, df, initial_level_pct, Q_MIN, Q_MAX, L_SP, L_MIN, L_MAX, DELTA_T_HOUR
            )
            
            if (count % 10 == 0) or (total_searches - count < 5) or (variance < min_variance):
                 print(f"  ({count}/{total_searches}) Kp={Kp:.3f}, Ti={Ti:.3f} ... 분산: {variance:.2f}")
            
            if variance < min_variance:
                min_variance = variance
                best_params = {'Kp': Kp, 'Ti': Ti}
                best_q_out_sim = q_out_sim_result
                best_level_sim = level_sim_result 

    # 4. 결과 분석
    print("\n[2단계] --- 최적화 결과 ---")
    if min_variance == float('inf'):
        print(f"탐색 실패: 이 탐색 범위에서 모든 조합이 레벨 제약({L_MIN}~{L_MAX}%)을 위반했습니다.")
        return
    
    improvement = ((BENCHMARK_VARIANCE - min_variance) / BENCHMARK_VARIANCE) * 100
    print(f"최적의 Kp: {best_params['Kp']:.3f}")
    print(f"최적의 Ti (적분 시간, 분): {best_params['Ti']:.3f}") 
    print(f"현재 분산 (Benchmark): {BENCHMARK_VARIANCE:.4f}") 
    print(f"최적 분산 (Simulated): {min_variance:.4f}")
    
    if improvement > 0:
        print(f"=> 유출량 산포(분산) 약 {improvement:.2f}% '감소' 예상")
    else:
        print(f"=> 유출량 산포(분산) 약 {-improvement:.2f}% '증가' 예상 (레벨 제약 비용)")
            
    # 5. 최종 그래프 저장
    if best_q_out_sim is not None:
        print("\n[2단계] 시뮬레이션 비교 그래프를 저장합니다...")
        
        if not os.path.exists(PLOT_OUTPUT_FOLDER):
            os.makedirs(PLOT_OUTPUT_FOLDER)
            
        date_str = df.index[0].strftime('%Y%m%d')
        
        try:
            base_plot_name = f"{SITE}_CALC_anti_{date_str}_{RESAMPLE_FREQ}_Ti_L{L_MIN}-{L_MAX}"
            
            # 그래프 1: 시계열
            fig1 = plt.figure(figsize=(15, 7))
            plt.plot(df.index, df[Q_OUT_COL], label=f'원본 유출량 (분산: {BENCHMARK_VARIANCE:.2f})',color='dimgray', alpha=0.7)
            plt.plot(df.index, best_q_out_sim, label=f'최적 유출량 (Kp={best_params["Kp"]:.2f}, Ti={best_params["Ti"]:.1f}, 분산: {min_variance:.2f})', color='green', linewidth=2)
            plt.title(f'유출량(Q_out) 시계열', fontsize=16)
            plt.legend(); plt.grid(True); plt.tight_layout()
            save_path1 = os.path.join(PLOT_OUTPUT_FOLDER, base_plot_name + "_Q_out.png")
            fig1.savefig(save_path1)
            plt.close(fig1)
            print(f"시계열 그래프 저장 완료: {save_path1}")
            
            # # 그래프 2: 히스토그램
            # fig2 = plt.figure(figsize=(10, 6))
            # plt.hist(df[Q_OUT_COL], bins=100, alpha=0.6, label=f'원본 유출량 (분산: {BENCHMARK_VARIANCE:.2f})', density=True)
            # plt.hist(best_q_out_sim, bins=100, alpha=0.6, label=f'최적 유출량 (분산: {min_variance:.2f})', density=True)
            # plt.title(f'유출량 산포(분산) 비교 (레벨 {L_MIN}~{L_MAX}%)', fontsize=16)
            # plt.legend(); plt.grid(True); plt.tight_layout()
            # save_path2 = os.path.join(PLOT_OUTPUT_FOLDER, "plot_histogram" + base_plot_name)
            # fig2.savefig(save_path2)
            # plt.close(fig2)
            # print(f"산포 그래프 저장 완료: {save_path2}")

            # 그래프 3: 시뮬레이션된 레벨
            fig3 = plt.figure(figsize=(15, 7))
            plt.plot(df.index, best_level_sim, label=f'시뮬레이션 레벨 (Kp={best_params["Kp"]:.2f}, Ti={best_params["Ti"]:.3f})', color='blue')
            plt.axhline(y=L_MAX, color='red', linestyle=':', linewidth=2, label=f'상한 (L_MAX = {L_MAX:.0f}%)')
            plt.axhline(y=L_SP, color='gray', linestyle=':', linewidth=1, label=f'목표 (SP = {L_SP:.0f}%)')
            plt.axhline(y=L_MIN, color='red', linestyle=':', linewidth=2, label=f'하한 (L_MIN = {L_MIN:.0f}%)')
            plt.plot(df.index, df['Level_total_pct'], label=f'과거 실제 레벨 (Min: {data_min_level:.1f}%)', color='darkgray', alpha=0.7)
            plt.title(f'집수조 Level', fontsize=16)
            plt.legend(); plt.grid(True); plt.tight_layout()
            save_path3 = os.path.join(PLOT_OUTPUT_FOLDER, base_plot_name + "_Level_sim.png")
            fig3.savefig(save_path3)
            plt.close(fig3)
            print(f"레벨 그래프 저장 완료: {save_path3}")
            
        except Exception as e:
            print(f"[2단계] 오류: 그래프 저장 실패. {e}")

# ======================================================================
# --- 스크립트 메인 실행 ---
# ======================================================================
if __name__ == "__main__":
    main()