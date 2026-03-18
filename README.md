# ERP-42 Autonomous Perception System

본 리포지토리는 자율주행 플랫폼 ERP-42 환경에서 Camera와 3D LiDAR 센서를 융합하여 구현한 ROS2 기반의 인지(Perception) 파이프라인입니다. 
실제 주행 환경에서 발생하는 조명 변화, 객체 가림, 센서 노이즈 등의 변수를 극복하기 위해, 전통적인 영상 처리 알고리즘부터 최신 딥러닝 모델까지 도입하며 시스템을 최적화한 문제 해결 과정을 담고 있습니다.

## Hardware & Software Environment

- Platform: ERP-42
- Sensors: 3D LiDAR, Web Camera, IMU, GPS
- OS/Framework: Ubuntu 22.04 / ROS2 Humble
- Deep Learning: PyTorch, YOLOv8, SCNN

## System Architecture

센서 데이터 취득부터 최종 인지 결과 발행까지의 주요 ROS2 Topic 흐름입니다.

### 1. Camera Pipeline
- Sub: `/camera1/image_raw` -> Node: `[Lane_Detector]` -> Pub: `/lane_detection/result`
- Sub: `/camera1/image_raw` -> Node: `[Traffic_Light_Detector]` -> Pub: `/traffic_light_signal`

### 2. LiDAR Pipeline
- Sub: `/velodyne_points` -> Node: `[CropBox_Filter]` -> Pub: `/cropbox_filtered`
- Sub: `/cropbox_filtered` + `/localization/kinematic_state` -> Node: `[Dynamic_Obstacle_Tracker]` -> Pub: `/accurate_dynamic_obstacles`

## Core Modules & Troubleshooting

각 파트별 디렉토리로 이동하면 버전별 상세한 발전 과정(History)과 핵심 알고리즘 코드를 확인할 수 있습니다.

### 1. Lane Detection (차선 인식)
- Stack: OpenCV, PyTorch, SCNN
- Summary: 그림자 및 장애물 가림 현상에 취약한 기존 OpenCV 및 RANSAC 방식의 한계를 극복하기 위해 공간적 특징 추출에 특화된 SCNN 모델을 도입했습니다. 한국형 SDLane 데이터셋으로 학습하여 가려진 차선의 곡률까지 추론하는 강건한 파이프라인을 구축했습니다.
- Link: [Lane_Detection 상세 보기](./Lane_Detection)

### 2. Traffic Light & Sign Recognition (신호등 및 표지판 인식)
- Stack: PyTorch, YOLOv8
- Summary: 초기 'YOLO 영역 검출 후 HSV 색상 판별' 방식에서 발생하는 원거리 인식률 저하 및 좌회전 신호 판독 불가 문제를 해결했습니다. AI Hub 한국 신호등 데이터셋을 활용하여 신호 상태 자체를 클래스로 분류하는 End-to-End 객체 인식 모델로 전면 개편하여 안정성을 확보했습니다.
- Link: [Traffic_Light_Sign 상세 보기](./Traffic_Light_Sign)

### 3. Dynamic Obstacle Detection (동적 장애물 판별)
- Stack: DBSCAN, Ego-motion Compensation, Scipy
- Summary: 정지된 객체가 자차의 이동으로 인해 동적 장애물로 오인식되는 문제를 해결했습니다. Odometry 데이터를 바탕으로 자차의 이동 변위 및 회전 각속도를 수학적으로 상쇄(Compensation)하여, 객체의 실제 절대 속도 벡터를 산출하는 고정밀 추적 알고리즘을 구현했습니다.
- Link: [Dynamic_Obstacle 상세 보기](./Dynamic_Obstacle)

## Repository Structure

```text
ERP42-Autonomous-Perception/
├── README.md                  # 전체 프로젝트 요약 및 아키텍처
├── Lane_Detection/            # 차선 인식 모델 최적화 히스토리 (v1~v3)
├── Traffic_Light_Sign/        # 신호등 및 표지판 인식 최적화 히스토리 (v1~v2)
└── Dynamic_Obstacle/          # 라이다 기반 장애물 판별 및 추적 알고리즘
```

## Demo & Execution

본 프로젝트의 코드는 ERP-42 실차 센서 환경과 ROS2 토픽에 강하게 결합되어 있습니다. 
실제 테스트에 사용된 대용량 센서 데이터(ROS2 Bag 파일)는 용량 제한으로 인해 리포지토리에 포함하지 않았으며, 각 모듈의 실제 구동 화면은 하위 디렉토리의 README에 첨부된 데모 영상(GIF)을 통해 확인하실 수 있습니다.

각 모듈의 코드를 로컬에서 검토하고자 할 경우, 다음 명령어를 통해 의존성을 설치할 수 있습니다.
```bash
pip install torch torchvision opencv-python numpy scipy scikit-learn ultralytics cv-bridge
```
