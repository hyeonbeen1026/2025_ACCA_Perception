# Traffic Light Recognition System (신호등 인식 시스템 진화 과정)

본 디렉토리는 자율주행 플랫폼(ERP-42)을 위한 신호등 인식 알고리즘의 발전 과정을 담고 있습니다. 
객체 검출 후 색상 필터링을 거치는 2단계 방식의 한계를 극복하고, 다양한 주행 환경(거리, 조명 변화, 좌회전 신호 등)에 대응하기 위해 End-to-End 딥러닝 분류 방식으로 파이프라인을 전면 개편한 최적화 과정을 기록했습니다.

<img width="716" height="291" alt="image" src="https://github.com/user-attachments/assets/6f4814f4-4084-4edc-b1bc-40aa37170b7f" />

## Version History & Troubleshooting

### [v1] YOLO (ROI Detection) + HSV Filter (초기 모델)
- File: `v1_TrafficLight_Hsv.py`
- Method: YOLO 모델로 신호등의 전체 영역(ROI)을 우선 검출한 뒤, 해당 영역을 HSV 색상 공간으로 변환하여 빨강, 노랑, 초록 픽셀 분포를 계산해 현재 신호를 판별합니다.
- Troubleshooting: 
  원거리 인식률이 크게 떨어지고, 카메라 각도나 역광에 의해 신호등 불빛이 명확하지 않을 때 색상 필터가 오판독을 일으키는 문제가 발생했습니다. 특히 '좌회전(화살표)' 신호를 구조적으로 판별할 수 없는 치명적인 한계가 있어 알고리즘 방식의 전면 수정이 필요했습니다.

### [v2] YOLOv8 End-to-End Classification (최종 채택)
- File: `v2_TrafficLight_Yolo.py`
- Method: AI Hub의 대규모 한국형 신호등 데이터를 활용하여, 신호등의 상태(빨강, 노랑, 초록, 좌회전 등) 자체를 개별 YOLO 클래스로 분리하여 학습시켰습니다. 세로형 오인식을 막기 위해 가로형 신호등만 필터링하고, 여러 신호등이 잡힐 경우 인식된 객체 중 가장 면적이 큰(차량과 가장 가까운) 신호를 최종 결과로 채택합니다.
- Troubleshooting: 
  HSV 색상 필터링이라는 불안정한 후처리 과정을 제거하여 처리 속도를 높이고 원거리 및 조명 변화에 대한 강건성을 확보했습니다. 또한 YOLOv8부터 v12까지의 자체 벤치마크 테스트를 진행한 결과, 인식률과 오인식률 측면에서 가장 안정적인 성능을 보여준 YOLOv8을 모델로 최종 채택했습니다. 

## How to Run (최종 v2 기준)

본 코드는 ROS2 Humble 및 PyTorch(CUDA) 환경에서 동작합니다.

### 1. Dependencies

```bash
pip install torch torchvision opencv-python numpy ultralytics cv-bridge
```

### 2. Execution

Custom 학습된 YOLOv8 가중치 파일(`best.pt`)이 지정된 경로에 존재하는지 확인 후 아래 명령어를 통해 노드를 실행합니다.

```bash
ros2 run traffic_light_detection traffic_light_node
```

### 3. I/O Topics

- Sub: `/camera1/image_raw` (sensor_msgs/Image)
- Pub (Signal): `/traffic_light_signal` (std_msgs/String) - 최종 판별된 신호 상태 발행
- Pub (Debug): `/traffic_light_debug` (sensor_msgs/Image) - Bounding Box가 그려진 시각화 이미지 발행
  
<img width="208" height="208" alt="image1" src="https://github.com/user-attachments/assets/ad030294-85dd-4553-aec1-aec9ff8516bc" />           <img width="208" height="208" alt="image2" src="https://github.com/user-attachments/assets/b1f1bf56-d487-4bc4-abfa-d7eb38745d9f" />
