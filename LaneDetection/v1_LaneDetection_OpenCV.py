import rclpy  
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.bridge = CvBridge()
        self.video_path = '/home/hyeon/Downloads/drive.mp4'
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.get_logger().error(f"Error opening video file {self.video_path}")
            return

        self.publisher = self.create_publisher(Image, '/lane_image', 10)
        self.timer = self.create_timer(1/30, self.process_frame)

        # 칼만 필터 초기화 (x, y, dx, dy)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], 
                                                 [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info('End of video')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        lane_image = self.detect_lane(frame)
        lane_msg = self.bridge.cv2_to_imgmsg(lane_image, 'bgr8')
        self.publisher.publish(lane_msg)

    def detect_lane(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        height, width = image.shape[:2]
        roi_vertices = np.array([[ 
            (100, height), (width - 100, height), (width - 450, height // 2), (450, height // 2)
        ]], dtype=np.int32)
        
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        roi_image = image.copy()
        cv2.polylines(roi_image, [roi_vertices], isClosed=True, color=(0, 255, 0), thickness=2)
        
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 50, minLineLength=50, maxLineGap=200)
        
        if lines is not None:
            left_lines, right_lines = self.filter_lanes_by_slope(lines)
            left_lane = self.ransac_lane_fitting(left_lines, height)
            right_lane = self.ransac_lane_fitting(right_lines, height)

            left_lane = self.extend_lane(left_lane, height)
            right_lane = self.extend_lane(right_lane, height)
            
            # 칼만 필터 적용
            left_lane = self.apply_kalman_filter(left_lane)
            right_lane = self.apply_kalman_filter(right_lane)
        
            line_image = np.zeros_like(image)
            if left_lane:
                cv2.line(line_image, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (255, 0, 0), 8)
            if right_lane:
                cv2.line(line_image, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (255, 0, 0), 8)
            
            return cv2.addWeighted(roi_image, 0.8, line_image, 1, 1)
        
        return image  # 검출된 차선이 없을 경우 원본 반환

    def filter_lanes_by_slope(self, lines):
        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if slope < -0.3:  # 기울기 완화
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3:
                right_lines.append((x1, y1, x2, y2))
        return left_lines, right_lines

    def ransac_lane_fitting(self, lines, img_height):
        if len(lines) == 0:
            return None
        
        X, Y = [], []
        for x1, y1, x2, y2 in lines:
            X.extend([x1, x2])
            Y.extend([y1, y2])

        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y)

        ransac = RANSACRegressor()
        ransac.fit(X, Y)
        
        x_start, x_end = min(X)[0], max(X)[0]
        y_start, y_end = ransac.predict(np.array([[x_start], [x_end]])).astype(int)
        
        return (x_start, y_start, x_end, y_end)

    def extend_lane(self, lane, img_height):
        if lane is None:
            return None

        x1, y1, x2, y2 = lane
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            slope = 0.001
        else:
            slope = dy / dx

        y_bottom = img_height
        x_bottom = int(x1 + (y_bottom - y1) / slope)

        return (x_bottom, y_bottom, x1, y1)

    def apply_kalman_filter(self, lane):
        if lane is None:
            return None

        x1, y1, x2, y2 = lane
        measurement = np.array([[np.float32(x1)], [np.float32(y1)]])
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()

        x1, y1 = int(prediction[0]), int(prediction[1])
        x2, y2 = int(prediction[0] + 50), int(prediction[1] + 100)

        return (x1, y1, x2, y2)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()