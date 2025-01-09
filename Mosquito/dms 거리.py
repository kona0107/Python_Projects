import csv
from qgis.core import (
    QgsProject, QgsSpatialIndex, QgsFeature, QgsDistanceArea, QgsGeometry
)

# 레이어 로드
dms_layer = QgsProject.instance().mapLayersByName('dms지점')[0]
gonggong_layer = QgsProject.instance().mapLayersByName('공공기관 기상관측지점 현황(2022년 상반기)')[0]
jonggwan_layer = QgsProject.instance().mapLayersByName('종관방재')[0]

# 거리 계산 도구 초기화
distance_calc = QgsDistanceArea()
distance_calc.setEllipsoid('WGS84')

# 공공기관과 종관방재 피처를 저장하며 source 정보를 별도로 추가
all_features = []
for layer, source_type in [(gonggong_layer, '공공기관'), (jonggwan_layer, '종관방재')]:
    for feature in layer.getFeatures():
        all_features.append((feature, source_type))  # (피처, 소스 타입) 형태로 저장

# 공간 인덱스 생성
spatial_index = QgsSpatialIndex()
for feature, _ in all_features:
    spatial_index.addFeature(feature)

# 결과 저장
results = []

for dms_feature in dms_layer.getFeatures():
    dms_geom = dms_feature.geometry()
    dms_point = dms_geom.asPoint()  # 중심 좌표를 QgsPointXY로 변환
    nearest_ids = spatial_index.nearestNeighbor(dms_point, 3)  # 가장 가까운 3개
    nearest_features = [(f, s) for f, s in all_features if f.id() in nearest_ids]

    # 거리 계산 및 정렬
    distances = []
    for feature, source_type in nearest_features:
        feature_geom = feature.geometry()
        distance = distance_calc.measureLine(dms_point, feature_geom.asPoint())
        distances.append((feature, source_type, distance))
    
    # 거리순 정렬
    distances.sort(key=lambda x: x[2])

    # 결과 저장
    row = [
        dms_feature['DMS_Code'], dms_feature['측정소']
    ]
    for feature, source_type, dist in distances[:3]:  # 가장 가까운 3개
        row.extend([
            feature['지점명'], feature['지점'] if source_type == '종관방재' else feature['표준지점번호'], dist, source_type
        ])
    results.append(row)

# 결과를 CSV 파일로 저장
output_file = 'F:/박정현/Mosquito/DMS_nearest_stations.csv'
with open(output_file, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow([
        'DMS CODE', 'DMS 관측소명',
        '기상 1순위 관측소명', '기상 1순위 관측소 코드', '기상 1순위 거리(m)', '기상 1순위 소스',
        '기상 2순위 관측소명', '기상 2순위 관측소 코드', '기상 2순위 거리(m)', '기상 2순위 소스',
        '기상 3순위 관측소명', '기상 3순위 관측소 코드', '기상 3순위 거리(m)', '기상 3순위 소스'
    ])
    writer.writerows(results)

print(f"결과가 '{output_file}'에 저장되었습니다.")
