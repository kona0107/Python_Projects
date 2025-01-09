import csv
from qgis.core import (
    QgsProject, QgsSpatialIndex, QgsFeature, QgsDistanceArea, QgsGeometry
)

# 레이어로드
dms_layer = QgsProject.instance().mapLayersByName('dms지점')[0]
jonggwan_layer = QgsProject.instance().mapLayersByName('종관방재')[0]

# 거리 계산 도구 초기화
distance_calc = QgsDistanceArea()
distance_calc.setEllipsoid('WGS84')

# 종관방재 피처만 저장
jonggwan_features = []
for feature in jonggwan_layer.getFeatures():
    jonggwan_features.append(feature)

# 공간 인덱스 생성
spatial_index = QgsSpatialIndex()
for feature in jonggwan_features:
    spatial_index.addFeature(feature)

# 결과 저장
results = []

for dms_feature in dms_layer.getFeatures():
    dms_geom = dms_feature.geometry()
    dms_point = dms_geom.asPoint()  # 중심 좌표를 QgsPointXY로 변환
    nearest_ids = spatial_index.nearestNeighbor(dms_point, 1)  # 가장 가까운 1개
    nearest_feature = [f for f in jonggwan_features if f.id() in nearest_ids][0]

    # 거리 계산
    feature_geom = nearest_feature.geometry()
    distance = distance_calc.measureLine(dms_point, feature_geom.asPoint())

    # 결과 저장
    row = [
        dms_feature['DMS_Code'], dms_feature['측정소'],  # DMS 지점 정보
        nearest_feature['지점명'], nearest_feature['지점'], distance  # 가장 가까운 종관방재 지점 정보
    ]
    results.append(row)

# 결과를 CSV 파일로 저장
output_file = 'F:/박정현/Mosquito/DMS_nearest_종관방재.csv'
with open(output_file, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow([
        'DMS_Code', '측정소',
        '가장 가까운 종관방재 관측소명', '종관방재 관측소 코드', '거리(m)'
    ])
    writer.writerows(results)

print(f"결과가 '{output_file}'에 저장되었습니다.")_