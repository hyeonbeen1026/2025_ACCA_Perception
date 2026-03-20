# Lane Detection Model Training & Benchmark

본 디렉토리는 자율주행 플랫폼(ERP-42)에 탑재할 최적의 차선 인식 딥러닝 모델을 선정하기 위해, 한국형 차선 데이터셋(42Dot SDLane)을 기반으로 여러 SOTA(State-of-the-Art) 모델을 직접 학습하고 평가한 벤치마크 과정을 담고 있습니다.

## Project Overview
- **File:** `SDLane.ipynb` (데이터 전처리, 3종 모델 학습, 평가 및 시각화, TensorRT 최적화 코드가 모두 포함된 통합 파이프라인)
- **Dataset:** 42Dot SDLane Dataset (국내 도로 환경에 최적화된 차선 데이터)
- **Environment:** Kaggle Notebook (GPU: Tesla T4 / P100)

## Models Evaluated

본 실험에서는 차선 인식에 주로 활용되는 3가지 구조의 모델을 구현하여 비교했습니다.
1. **ERFNet:** Dilated Convolution(팽창 컨볼루션) 기반의 1D Non-Bottleneck 구조를 사용하여 연산량을 줄인 경량화 모델.
2. **LaneNet:** VGG16 Backbone 기반으로 Binary Segmentation과 Instance Embedding을 동시에 수행하여 개별 차선을 구분하는 모델 (Discriminative Loss 적용).
3. **SCNN (Spatial CNN) :** ResNet34 Backbone 특징맵에 방향성 컨볼루션을 적용하여, 차선이 장애물에 가려져 보이지 않더라도 공간적 연속성을 기반으로 궤적을 추론하는 모델.

## Benchmark Results

동일한 SDLane 데이터셋 및 Albumentations 증강(GridDropout, ColorJitter 등) 환경에서 5회 학습(5 Runs)을 진행한 평균 성능 지표입니다.

| Model | Backbone | IoU | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| ERFNet | Custom | 0.323 | 0.324 | 0.986 | 0.487 |
| LaneNet | VGG16 | 0.444 | 0.739 | 0.527 | 0.614 |
| **SCNN** | ResNet34 | **0.491** | 0.493 | **0.989** | **0.656** |

> **Conclusion:** > 평가 결과 SCNN 모델이 가장 높은 IoU(0.491)와 F1-Score(0.656)를 기록했습니다. 특히 가림 현상이 잦은 주행 환경에서 객체 뒤의 차선을 추론하는 공간적 특성 파악 능력이 가장 뛰어나, 최종 인지 파이프라인의 차선 추론 모델로 **SCNN을 채택**하였습니다.

## Edge Device Optimization (TensorRT)

ERP-42 차량 내부의 연산 장치(Jetson 등)에서 실시간(Real-time) 추론을 수행하기 위해, 최종 채택된 SCNN 모델을 최적화하는 과정을 거쳤습니다.

1. **ONNX Export:** PyTorch(`.pth`) 모델을 프레임워크 독립적인 `.onnx` 포맷으로 변환. (Dynamic Batch Size 적용)
2. **TensorRT Engine Build:** 자율주행 엣지 보드에서의 추론 속도(FPS)를 극대화하기 위해, FP16 정밀도(Precision)를 적용하여 TensorRT 엔진(`.trt`)으로 직렬화(Serialization) 완료.

## How to Run

1. Kaggle 환경에서 새 노트북을 생성하고 `SDLane.ipynb` 파일을 Import 합니다.
2. 데이터셋 경로 설정: Kaggle에 `42Dot SDLane` 데이터셋을 Add Data한 후, 코드 내 `image_dir` 및 `label_dir` 경로를 본인의 Kaggle 환경에 맞게 수정합니다.
3. 실행 방식: 코드 최하단의 `test_option` 변수를 `"single"`(단일 이미지 시각화) 또는 `"loader"`(배치 단위 검증)로 설정하여 벤치마크 결과를 확인할 수 있습니다.
