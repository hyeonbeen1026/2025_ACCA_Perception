# Robust Lane Detection System (차선 인식 시스템 진화 과정)

본 디렉토리는 자율주행 플랫폼(ERP-42)의 안정적인 차선 유지를 위해 개발된 차선 인식(Lane Detection) 알고리즘의 발전 과정을 담고 있습니다. 
단순 영상 처리에서 시작하여, 실제 주행 환경의 다양한 변수(그림자, 가림 현상 등)를 극복하기 위해 SOTA 딥러닝 모델로 구조를 개편한 3단계의 최적화 과정을 기록했습니다.

## Version History & Troubleshooting

### [v1] OpenCV + RANSAC & Kalman Filter (초기 모델)
- File: `v1_LaneDetection_OpenCV.py`
- Method: Grayscale 변환, Gaussian Blur, Canny Edge로 엣지를 추출한 뒤 ROI 영역 내에서 HoughLinesP를 사용하여 차선을 검출합니다.
- Troubleshooting: 
  그림자나 장애물이 차선을 가릴 경우 오인식하는 문제가 발생했습니다. 노이즈 제거와 연속성 확보를 위해 RANSAC과 Kalman Filter를 도입했으나, 차량 가림이나 차선 변경 시 노이즈가 증가하면 예측값이 크게 튀는 한계가 있어 딥러닝 방식으로 전환을 결정했습니다.

### [v2] YOLO Segmentation (딥러닝 도입)
- File: `v2_LaneDetection_YOLO.py`
- Method: Labelme를 활용해 직접 라벨링한 시내 주행 영상(60장)으로 YOLO 모델을 학습시켜 Segmentation Mask를 추출합니다.
- Troubleshooting: 
  기존 방식 대비 잔노이즈에 의한 오인식은 감소했습니다. 하지만 장애물이 차선을 완전히 가린 경우, 모델이 해당 영역을 객체가 없다고 판단하여 가려진 차선을 전혀 추론하지 못하는 미인식 문제가 발생했습니다. 이를 해결하기 위해 차선의 공간적 연속성을 이해하는 모델의 필요성을 확인했습니다.

### [v3] SCNN + SDLane Dataset (최종 채택)
- File: `v3_LaneDetection_SCNN.py`
- Method: 차선처럼 얇고 긴 형태의 공간적 특징을 파악하는 데 특화된 SCNN(Spatial CNN) 구조를 도입했습니다. 또한, 한국 도로 환경에 적합한 42Dot SDLane 데이터셋을 발굴하여 모델을 학습시켰습니다.
- Results: 
  차선이 차량이나 장애물에 가려져도 주변 픽셀 정보를 바탕으로 차선의 곡률과 궤적을 성공적으로 추론합니다. ROS2 환경에서 `/camera1/image_raw` 토픽을 실시간으로 구독하고 GPU 기반 추론 후 `/lane_detection/result` 토픽을 발행하는 파이프라인 구축을 완료했습니다.

## How to Run (최종 v3 기준)

본 코드는 ROS2 Humble 및 PyTorch(CUDA) 환경에서 동작합니다.

### 1. Dependencies

```bash
pip install torch torchvision opencv-python numpy
```

### 2. Execution

`SDLane_SCNN.pth` 가중치 파일이 경로에 존재하는지 확인 후 아래 명령어를 통해 노드를 실행합니다.

```bash
ros2 run lane_detection lane_detector
```

### 3. I/O Topics

- Sub: `/camera1/image_raw` (sensor_msgs/Image)
- Pub: `/lane_detection/result` (sensor_msgs/Image)
  
![KakaoTalk_20260318_201058534](https://github.com/user-attachments/assets/d17658fe-f16d-4619-9351-17fa4cd943c4)
