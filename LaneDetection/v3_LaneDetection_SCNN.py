
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import functional as TF
import torchvision.models as models
import warnings
 
warnings.filterwarnings("ignore", category=UserWarning)


class SCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Backbone (ResNet34)
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights=None)
        self.encoder1 = nn.Sequential(*list(backbone.children())[:5])  # /2
        self.encoder2 = backbone.layer2  # /4
        self.encoder3 = backbone.layer3  # /8
        self.encoder4 = backbone.layer4  # /16

        # SCNN 모듈 (패딩 추가)
        self.scnn = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1,9), padding=(0,4)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(9,1), padding=(4,0)),
            nn.ReLU()
        )

        # 디코더 (크기 조정 보장)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, num_classes, kernel_size=1)  # 최종 출력
        )

    def forward(self, x):
        # 인코더
        e1 = self.encoder1(x)  # /2
        e2 = self.encoder2(e1) # /4
        e3 = self.encoder3(e2) # /8
        e4 = self.encoder4(e3) # /16
        
        # SCNN
        x = self.scnn(e4)
        
        # 디코더 (크기 자동 조정)
        x = self.decoder(x)
        return x
class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
        
        # 모델 로드 (CPU로 명시적으로 로드)
        self.model = SCNN(num_classes=2)
        try:
            # 모델 경로를 절대 경로로 지정 (필요에 따라 수정)
            model_path = '/home/hyeon/ros2_ws/src/lane_detection/lane_detection/SDLane_SCNN.pth'
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.get_logger().info("Model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            raise
        
        # 이미지 전처리 파라미터
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(self.device)
        
        # ROS2 설정
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera1/image_raw',  # 카메라 토픽 (필요에 따라 변경)
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Image, '/lane_detection/result', 10)
        
        self.get_logger().info("Lane Detector initialized")

    def preprocess(self, cv_image):
        # 이미지 전처리
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 608))  # 모델 입력 크기에 맞춤
        
        # 텐서 변환 및 정규화
        image = TF.to_tensor(image).to(self.device)
        image = (image - self.mean) / self.std
        image = image.unsqueeze(0)  # 배치 차원 추가
        
        return image

    def postprocess(self, output, original_size):
        # 예측 결과 후처리
        output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=True)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        # 차선 마스크 생성 (차선은 1, 배경은 0)
        lane_mask = (pred_mask == 1).astype(np.uint8) * 255
        
        # 원본 이미지 크기로 리사이즈
        lane_mask = cv2.resize(lane_mask, (original_size[1], original_size[0]))
        
        return lane_mask

    def image_callback(self, msg):
        try:
            # ROS 이미지를 OpenCV 형식으로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            original_size = cv_image.shape[:2]
            
            # 전처리
            input_tensor = self.preprocess(cv_image)
            
            # 추론
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # 후처리
            lane_mask = self.postprocess(output, original_size)
            
            # 차선 영역을 원본 이미지에 오버레이
            overlay = cv_image.copy()
            overlay[lane_mask == 255] = [0, 255, 0]  # 차선 영역을 초록색으로 표시
            
            # 결과 이미지 발행
            result_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            self.publisher.publish(result_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in image processing: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    lane_detector = LaneDetector()
    rclpy.spin(lane_detector)
    lane_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
