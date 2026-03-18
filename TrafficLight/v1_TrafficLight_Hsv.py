import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

# YOLOv8 모델 로드
MODEL_PATH = '/home/hyeon/traffic/runs/detect/train2/weights/best.pt'
model = YOLO(MODEL_PATH)

# HSV 색상 범위 정의
COLOR_RANGES = {
    "red": [(0, 100, 100), (10, 255, 255)],
    "yellow": [(20, 100, 100), (30, 255, 255)],
    "green": [(40, 50, 50), (90, 255, 255)]
}

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')
        self.publisher = self.create_publisher(String, 'traffic_light_color', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.process_frame)
        self.cap = cv2.VideoCapture('/home/hyeon/Downloads/drive.mp4')

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video stream")
            return

        # YOLOv8 추론 수행
        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()  # 바운딩 박스 정보 가져오기
        
        for detection in detections:
            x1, y1, x2, y2, _, class_id = map(int, detection)
            if class_id == 9:  # YOLOv8의 'traffic light' 클래스 ID
                roi = frame[y1:y2, x1:x2]
                detected_color = self.detect_color(roi)
                
                # 결과 퍼블리시
                msg = String()
                msg.data = detected_color
                self.publisher.publish(msg)
                self.get_logger().info(f'Detected Traffic Light Color: {detected_color}')

    def detect_color(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        max_color = "unknown"
        max_pixels = 0
        
        for color, (lower, upper) in COLOR_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            pixels = cv2.countNonZero(mask)
            if pixels > max_pixels:
                max_pixels = pixels
                max_color = color
                
        return max_color


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
