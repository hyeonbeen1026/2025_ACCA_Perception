import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YOLOSegmentationNode(Node):
    def __init__(self):
        super().__init__('yolo_segmentation_node')

        # YOLO 모델 로드
        self.model = YOLO("/home/hyeon/drive/runs/segment/train/weights/best.pt")
        self.bridge = CvBridge()

        # ROS 2 토픽 발행
        self.segmentation_publisher = self.create_publisher(String, '/yolo/segmentation_masks', 10)
        self.image_publisher = self.create_publisher(Image, '/yolo_cam', 10)

        # 동영상 파일 경로 설정
        self.video_path = "/home/hyeon/Downloads/drive.mp4" 
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video: {self.video_path}")
            return

        self.get_logger().info("YOLO Segmentation Node Initialized")

        # 타이머 설정 (주기적으로 프레임 처리)
        self.timer = self.create_timer(0.03, self.process_frame)  

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video reached or failed to read frame.")
            self.cap.release()
            self.destroy_node()
            return

        try:
            # YOLO 세그멘테이션 실행
            results = self.model.predict(source=frame, save=False, conf=0.5, show=False)

            # 결과 처리 및 마스크 발행
            for result in results:
                masks = result.masks
                if masks is not None:
                    for mask in masks.xy:  # 마스크 좌표
                        mask_json = {"mask": mask.tolist()}
                        self.segmentation_publisher.publish(String(data=str(mask_json)))

            # YOLO가 그린 프레임을 발행
            annotated_frame = results[0].plot()  # YOLO가 그린 프레임
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.image_publisher.publish(annotated_msg)

            # 시각화
            cv2.imshow("YOLO Segmentation", annotated_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    yolo_segmentation_node = YOLOSegmentationNode()

    try:
        rclpy.spin(yolo_segmentation_node)
    except KeyboardInterrupt:
        pass

    yolo_segmentation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()