# LiDAR-based Dynamic Obstacle Detection (라이다 기반 동적/정적 장애물 판별 시스템)

본 디렉토리는 3D LiDAR 데이터와 차량의 Odometry 정보를 융합하여, 주행 환경 내의 장애물을 탐지하고 '동적(Dynamic)' 장애물과 '정적(Static)' 장애물로 분류하는 추적(Tracking) 알고리즘을 담고 있습니다. 
자율주행 차량 스스로의 움직임(Ego-motion)을 상쇄하여 객체의 실제 절대 속도를 정확하게 추정하는 것이 본 알고리즘의 핵심입니다.

## Core Architecture & Troubleshooting

### 1. 자율주행 차량의 움직임 보상 (Ego-motion Compensation)
- Method: 자율주행 차량이 이동 중일 때는 정지된 가로수나 주차된 차량도 센서상에서는 상대 속도를 가지게 되어 동적 장애물로 오인식될 수 있습니다. 이를 해결하기 위해 `/localization/kinematic_state` 토픽에서 차량의 선속도와 각속도를 수신합니다. 회전 행렬(Rotation Matrix)과 각속도 보상 수식을 적용하여 이동 변위를 계산함으로써 장애물의 실제 절대 속도 벡터를 도출했습니다.

### 2. DBSCAN 군집화 및 객체 추적 (Clustering & Tracking)
- Method: 라이다 포인트 클라우드의 노이즈를 줄이기 위해 Z축 높이 필터링을 거친 후, 밀도 기반 군집화 알고리즘인 DBSCAN을 사용하여 객체를 분리합니다. 이후 프레임 단위로 유클리디안 거리 기반의 Nearest Neighbor 매칭을 통해 동일 객체를 추적합니다.

### 3. 속도 필터링 및 안정성 확보 (Velocity Filtering)
- Troubleshooting: 일시적인 센서 노이즈나 매칭 오류로 인해 객체의 속도가 비정상적으로 튀는(Spike) 문제가 발생했습니다. 
- Solution: 이를 방지하기 위해 추적 중인 객체의 이전 속도와 현재 속도를 가중 평균하는 필터(Alpha filter)를 적용하고, 물리적으로 불가능한 가속도(2G 이상)를 제한하는 로직을 추가했습니다. 또한 큐(Deque)를 활용한 속도 히스토리의 중앙값과 평균값을 결합하여, 일정 프레임 이상 지정된 속도를 유지할 때만 '동적(Dynamic)'으로 확정 짓도록 안정성을 극대화했습니다.

## How to Run 

본 코드는 ROS2 Humble 및 Python3 환경에서 동작하며, 벡터 연산 및 군집화를 위한 외부 라이브러리가 필요합니다.

### 1. Dependencies

```bash
pip install numpy scipy scikit-learn
```

### 2. Execution

ROS2 Workspace 빌드 후 아래 명령어를 통해 노드를 실행합니다.

```bash
ros2 run obstacle_detection dynamic_obstacle_node
```

### 3. I/O Topics & Visualization

- Sub (LiDAR): `/cropbox_filtered` (sensor_msgs/PointCloud2)
- Sub (Odom): `/localization/kinematic_state` (nav_msgs/Odometry)
- Pub (Markers): `/accurate_dynamic_obstacles` (visualization_msgs/MarkerArray)
  - Red Cube: 동적 장애물 (속도 벡터 Arrow 포함)
  - Blue Cube: 정적 장애물
- Pub (Debug): `/obstacle_debug` (std_msgs/String) - 객체별 속도 및 상태 텍스트 출력
