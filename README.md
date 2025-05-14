# 제조 불량률 감소 목적 자동화 검수 시스템

YOLO + DeepSORT 기반의 실시간 검수 자동화 솔루션을 개발하여, 제조 현장의 검수 피로도와 불량률을 효과적으로 감소시킨 프로젝트입니다.

## 개요

- 기간: 2022.10 ~ 2022.12 (3개월)
- 기관: AGP
- 역할: 기획 / 데이터 수집 / 모델 설계 및 학습 / GUI 개발
- 기술 스택: Python, PyTorch, YOLOv5, DeepSORT, OpenCV, PyQt5


## 프로젝트 배경

- 맞춤형 제품의 조합이 증가하며 검수 과정에서 피로도가 급증
- 조합 실수로 인해 불량률 및 고객 CS 증가
- 제조 환경은 네트워크 제한이 있어 웹 기반 시스템 도입 불가


## 문제 정의

- 옵션 조합 폭발 → 검수 단계 증가 → 피로도 상승
- 객체 탐지 정확도 낮음 (조도, 유사도, 라벨 오류)
- 비개발자 대상 → 사용이 쉬운 UI 필요


## 해결 방안 개요

| 문제 상황 | 해결 솔루션 |
|-----------|--------------|
| 조합 폭발, 피로도 증가 | YOLO로 제품 탐지, DeepSORT로 옵션 추적하여 자동 카운트 |
| 탐지 정확도 저하 | 실데이터 기반 재수집 + 수작업 라벨링으로 정확도 향상 |
| 사용 어려움 | PyQt5 기반 GUI + 설치형 프로그램으로 배포 |


## 시스템 구조
### 전체 구조
![Image](https://github.com/user-attachments/assets/ec3dfe22-39a2-425b-a663-81b99fbe7383)

### 세부 구조
![Image](https://github.com/user-attachments/assets/6d3c4dc6-efea-4dd3-8fb5-8cb2f4b54f15)

- YOLO: 제품 탐지
- DeepSORT: 객체 ID 추적
- OJB(판단 기준선) 기준으로 개수 카운트
- 검수 결과 GUI 실시간 표시

## 데이터 수집 및 전처리

![Image](https://github.com/user-attachments/assets/f7b63bd0-b2d5-4c96-962b-43bc57a7dc51)

- 직접 제조현장 방문, 50개 이상 옵션 영상 수집
- 약 400장의 BBox 수작업 어노테이션
- 조도 보정, 중복 제거, 클래스 불균형 조정


## 모델 학습 및 성능 개선

![image](https://github.com/user-attachments/assets/4d04ea37-b5e9-4e98-b796-0d6c08a4e605)

| 항목 | 초기 YOLO | 개선 후 YOLO |
|------|-----------|--------------|
| mAP@0.5 | 0.65 | **0.80** |

- 오탐 원인을 분석해 데이터셋 개선
- 수작업 라벨 검수 및 추가 수집
- 재학습을 통해 실시간 추론에서도 정확도 확보


## 추적 및 판단 로직

![image](https://github.com/user-attachments/assets/a5fe646e-13a4-4bbc-8ab7-3472ea0db201)

![image](https://github.com/user-attachments/assets/c801e6b5-7aff-48bd-a2e9-4be843de237c)

- 중심점 이동 경로 설정
- 조건 기준선 통과 시 → 카운트 트리거 발생
- 좌→우 방향 전용 필터링 조건 적용


## 🖥 GUI 및 설치형 프로그램 개발

![image](https://github.com/user-attachments/assets/0e0dd884-6ae9-49b5-9289-a21275ed31f7)

![image](https://github.com/user-attachments/assets/c7f97723-cfab-4f60-815e-61be728d875d)

- PyQt 기반 로컬 GUI
- 탐지 결과를 영상에 실시간 시각화
- 옵션명, 개수, 상태(OK/NG)를 표시
- 네트워크 없이도 사용 가능한 `.exe` 설치형으로 배포

## 프로젝트 성과

- 정확도 0.65 → 0.80 향상
- 현장 적용 시도 후 **작업 시간 단축 및 실수 감소 확인**
- 비개발자도 사용할 수 있도록 UX 최적화

## Reference
- [YOLOv5](https://github.com/ultralytics/yolov5) for Object Detection  
- [StrongSORT](https://github.com/dyhBUPT/StrongSORT) for Object Tracking  
