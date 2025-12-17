import bigdataquery as bdq
import datetime, time
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys 
sys.path.append('/config/work/OCAP_TRUE_FALSE/CODE/ALL')
from login import login_bdq

USER_ID = 'hgreen.auto'
SITE = "PT"
if SITE ==  'PT':
    PARARM_INDEX_LIST = [3,4,5,6,7,8,9,10,11,12] # 인덱스 예시 수정
elif SITE == 'HS':
    PARARM_INDEX_LIST = [13,14,15,16,17,18,19,20,21,22]


START_DT = datetime.datetime(2025, 11,15, 0, 0,0)
END_DT = datetime.datetime(2025, 11, 30, 0, 0, 0)
CHUNK_HOUR = 4 # 메모리 과부하 방지를 위한 청크 단위 (시간)

OUT_DIR = rf"/config/work/개인폴더/박정현/PID_control/RAW_DATA/{SITE}"

os.makedirs(OUT_DIR, exist_ok=True)

REQ_COLUMNS = ['param_index', 'act_time', 'param_value']
cur_start = START_DT
chunk_seq =1

while cur_start < END_DT:
    cur_end = min(cur_start + datetime.timedelta(hours=CHUNK_HOUR), END_DT)
    
    param = {
        "dataForm": cur_start.strftime('%Y-%m-%d %H:%M:%S'),
        "dataTo": cur_end.strftime('%Y-%m-%d %H:%M:%S'),
        "param_index": PARARM_INDEX_LIST,
        "site": "PT"
    }

    df= bdq.getTraceData(
        param=param,
        verbose=False,
        show_sql=False,
        data_name='iees_trace_data',
        user_name=USER_ID,
    )
    print(f"[{cur_start} ~ {cur_end}] rows: {len(df)}")

    col_info = bdq.gerColumnInfo('iees_trace_data')[['cols', 'array']]
    array_cols = [c for c, a in col_info.values.tolist()
                  if a and c in REQ_COLUMNS]
    unnested = df.apply(
        lambda s : s.explode() if s.name in array_cols else s
    ).reset_index(drop=True)   

    target = unnested[['pararm_index', 'act_time', 'param_value']]
    pivot = target.pivot_table(
        index='act_time',
        columns='param_index',
        values='param_value',
        aggfunc='first' # 필요에 따라 mean, sum 등으로 변경 가능
    )
    pivot.index = pd.to_datetime(pivot.index)   
    # 끝 초과 행 제거
    pivot = pivot[pivot.index < pd.Timestamp(cur_end)]

    # 파일명 예시 : PT_TRACE_RAW_DATA_0922_0001.csv
    file_name = (
        f"PT_TRACE_RAW_DATA_"
        f"{cur_start.strftime('%m%d')}_"
        f"{chunk_seq:04d}.csv"
    )
    file_path = os.path.join(OUT_DIR, file_name)
    pivot.to_csv(file_path, index=True)
    print(f"{file_name} 저장 완료 -> {file_path}")

    cur_start
    chunk_seq += 1

