import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, String
import tf2_ros
from tf2_ros import TransformException
from collections import deque
import math
from rclpy.time import Time

class AccurateDynamicObstacleDetector(Node):
    def __init__(self):
        super().__init__('accurate_dynamic_obstacle_detector')
        
        # Parameter configuration
        self.declare_parameters(namespace='',
            parameters=[
                ('cluster_eps', 0.5),
                ('cluster_min_samples', 5),
                ('min_dynamic_speed', 1.8),
                ('max_static_speed', 0.1),
                ('z_min', -0.5),
                ('z_max', 2.0),
                ('tracking_window', 5),
                ('required_consistent_frames', 3),
                ('transform_timeout', 0.1),
                ('base_frame', 'base_link'),
                ('lidar_frame', 'velodyne'),
                ('debug_mode', True),
                ('max_track_age', 1.0),
                ('min_track_length', 0.5),
                ('velocity_filter_alpha', 0.2),
                ('max_acceleration', 9.8 * 2),  # 2G acceleration limit
                ('min_confidence_frames', 3)    # Minimum frames for reliable detection
            ])
        
        # Subscribers
        self.sub_cloud = self.create_subscription(
            PointCloud2, '/cropbox_filtered', self.pointcloud_callback, 10)
        self.sub_odom = self.create_subscription(
            Odometry, '/localization/kinematic_state', self.odometry_callback, 10)
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/accurate_dynamic_obstacles', 10)
        self.debug_pub = self.create_publisher(String, '/obstacle_debug', 10)
        
        # TF Listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.time.Duration(seconds=10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # State variables
        self.vehicle_odom = None
        self.obstacle_tracks = {}
        self.obstacle_id_counter = 0
        self.last_odom_time = None
        self.last_position = np.zeros(3)
        self.last_orientation = np.array([0, 0, 0, 1])
        self.vehicle_velocity = np.zeros(3)
        self.vehicle_angular_velocity = np.zeros(3)

    def odometry_callback(self, msg):
        """Update vehicle speed and position"""
        self.vehicle_odom = msg
        self.last_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.last_orientation = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        self.vehicle_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        self.vehicle_angular_velocity = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])
        self.last_odom_time = msg.header.stamp

    def extract_points(self, cloud_msg):
        """Safely extract point cloud data"""
        try:
            points = []
            for p in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append([p[0], p[1], p[2]])
            return np.array(points, dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Point extraction failed: {str(e)}")
            return np.empty((0, 3), dtype=np.float32)

    def pointcloud_callback(self, msg):
        try:
            if self.vehicle_odom is None or self.last_odom_time is None:
                return

            # Time interval calculation (using ROS time)
            current_time = Time.from_msg(msg.header.stamp)
            last_time = Time.from_msg(self.last_odom_time)
            dt = (current_time - last_time).nanoseconds / 1e9
            if dt <= 0.001 or dt > 0.5:
                return

            # TF transformation
            try:
                lidar_to_base = self.tf_buffer.lookup_transform(
                    self.get_parameter('base_frame').value,
                    self.get_parameter('lidar_frame').value,
                    current_time,
                    timeout=rclpy.time.Duration(seconds=self.get_parameter('transform_timeout').value))
            except TransformException as e:
                self.get_logger().warn(f"TF error: {str(e)}", throttle_duration_sec=1)
                return

            # Point cloud processing
            points = self.extract_points(msg)
            if len(points) == 0:
                return

            # Height filtering
            z_filter = (points[:, 2] > self.get_parameter('z_min').value) & \
                       (points[:, 2] < self.get_parameter('z_max').value)
            filtered_points = points[z_filter]
            
            if len(filtered_points) < 5:
                return

            # Convert to base_link frame
            rotation = R.from_quat([
                lidar_to_base.transform.rotation.x,
                lidar_to_base.transform.rotation.y,
                lidar_to_base.transform.rotation.z,
                lidar_to_base.transform.rotation.w
            ])
            base_link_points = rotation.apply(filtered_points) + np.array([
                lidar_to_base.transform.translation.x,
                lidar_to_base.transform.translation.y,
                lidar_to_base.transform.translation.z
            ], dtype=np.float32)

            # Clustering
            clustering = DBSCAN(
                eps=self.get_parameter('cluster_eps').value,
                min_samples=self.get_parameter('cluster_min_samples').value
            ).fit(base_link_points)

            # Cluster extraction
            current_clusters = {}
            for label in np.unique(clustering.labels_):
                if label == -1:
                    continue
                cluster_mask = (clustering.labels_ == label)
                cluster_points = base_link_points[cluster_mask]
                if len(cluster_points) >= 5:
                    current_clusters[label] = {
                        'center': np.mean(cluster_points, axis=0),
                        'points': cluster_points,
                        'size': np.ptp(cluster_points, axis=0),
                        'time': current_time
                    }

            # Obstacle tracking
            self.update_tracks(current_clusters, current_time, dt)
            self.visualize_obstacles(msg.header)

        except Exception as e:
            self.get_logger().error(f"Main loop error: {str(e)}")

    def update_tracks(self, current_clusters, current_time, dt):
        """Obstacle tracking with improved velocity calculation"""
        if self.vehicle_odom is None:
            return

        debug_msg = String()
        matched_clusters = set()

        # Update existing tracks
        for track_id, track in list(self.obstacle_tracks.items()):
            # Check track age
            track_age = (current_time - Time.from_msg(track['last_update_time'])).nanoseconds / 1e9
            if track_age > self.get_parameter('max_track_age').value:
                del self.obstacle_tracks[track_id]
                continue

            # Predict position if frames were missed
            if track_age > dt * 1.5:  # More than 1.5x expected interval
                predicted_position = track['current_position'] + track['current_velocity'] * track_age
            else:
                predicted_position = track['current_position']

            # Find closest cluster
            best_match_label = None
            min_distance = float('inf')
            
            for label, cluster in current_clusters.items():
                if label in matched_clusters:
                    continue
                
                dist = np.linalg.norm(cluster['center'] - predicted_position)
                if dist < min_distance and dist < 1.0:
                    min_distance = dist
                    best_match_label = label
            
            if best_match_label is not None:
                matched_clusters.add(best_match_label)
                cluster = current_clusters[best_match_label]
                actual_dt = (current_time - Time.from_msg(track['last_update_time'])).nanoseconds / 1e9
                
                # Vehicle motion compensation
                rot_matrix = R.from_quat(self.last_orientation).as_matrix()
                angular_velocity = self.vehicle_angular_velocity
                displacement = cluster['center'] - track['current_position']
                
                # Angular velocity compensation
                rotation_compensation = np.cross(angular_velocity, displacement) * actual_dt
                
                # Calculate actual obstacle velocity
                raw_obstacle_velocity = (displacement / actual_dt) - (rot_matrix @ self.vehicle_velocity + rotation_compensation)
                
                # Enhanced velocity filtering
                alpha = self.get_parameter('velocity_filter_alpha').value
                filtered_velocity = alpha * raw_obstacle_velocity + (1 - alpha) * track['current_velocity']
                
                # Physical plausibility check
                max_acceleration = self.get_parameter('max_acceleration').value
                acceleration = np.linalg.norm(filtered_velocity - track['current_velocity']) / actual_dt
                if acceleration > max_acceleration:
                    # If acceleration exceeds limit, maintain previous velocity
                    filtered_velocity = track['current_velocity']
                
                obstacle_speed = np.linalg.norm(filtered_velocity[:2])  # 2D speed only
                
                # Update speed history (using moving average)
                track['speed_history'].append(obstacle_speed)
                if len(track['speed_history']) > self.get_parameter('tracking_window').value:
                    track['speed_history'].popleft()
                
                # Combined median and mean for robustness
                avg_speed = 0.7 * np.median(track['speed_history']) + 0.3 * np.mean(track['speed_history']) \
                            if track['speed_history'] else 0.0
                
                # Dynamic/static classification
                if avg_speed > self.get_parameter('min_dynamic_speed').value:
                    track['dynamic_frames'] += 1
                elif avg_speed < self.get_parameter('max_static_speed').value:
                    track['dynamic_frames'] = max(0, track['dynamic_frames'] - 1)
                
                # Calculate movement distance
                movement = np.linalg.norm(displacement)
                track['total_movement'] = track.get('total_movement', 0.0) + movement
                
                # Stability verification
                if len(track['speed_history']) > 1:
                    speed_changes = np.abs(np.diff(track['speed_history']))
                    if np.all(speed_changes < 1.0):  # Speed changes < 1m/s
                        track['stable_frames'] = track.get('stable_frames', 0) + 1
                    else:
                        track['stable_frames'] = max(0, track.get('stable_frames', 0) - 2)
                
                # Update track state
                track.update({
                    'current_position': cluster['center'],
                    'last_update_time': current_time.to_msg(),
                    'size': cluster['size'],
                    'is_dynamic': track['dynamic_frames'] >= self.get_parameter('required_consistent_frames').value,
                    'current_velocity': filtered_velocity,
                    'avg_speed': avg_speed,
                    'last_speed': obstacle_speed
                })
                
                if self.get_parameter('debug_mode').value:
                    debug_msg.data += (f"Track {track_id}: "
                                     f"Speed={avg_speed:.2f}m/s "
                                     f"Vehicle={np.linalg.norm(self.vehicle_velocity):.2f}m/s "
                                     f"Frames={track['dynamic_frames']} "
                                     f"Stable={track.get('stable_frames', 0)}\n")
        
        # Create new tracks for unmatched clusters
        for label, cluster in current_clusters.items():
            if label not in matched_clusters:
                self.obstacle_tracks[self.obstacle_id_counter] = {
                    'current_position': cluster['center'],
                    'last_update_time': current_time.to_msg(),
                    'size': cluster['size'],
                    'speed_history': deque([0.0], maxlen=self.get_parameter('tracking_window').value),
                    'dynamic_frames': 0,
                    'is_dynamic': False,
                    'current_velocity': np.zeros(3, dtype=np.float32),
                    'avg_speed': 0.0,
                    'total_movement': 0.0,
                    'last_speed': 0.0,
                    'stable_frames': 0
                }
                self.obstacle_id_counter += 1

        # Remove unstable tracks
        self.obstacle_tracks = {
            k: v for k, v in self.obstacle_tracks.items()
            if not (len(v['speed_history']) == self.get_parameter('tracking_window').value and 
                   v.get('stable_frames', 0) < 2)
        }

        if debug_msg.data:
            self.debug_pub.publish(debug_msg)

    def visualize_obstacles(self, header):
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header = header
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for track_id, track in self.obstacle_tracks.items():
            # Filter by minimum movement and stability
            if (track.get('total_movement', 0.0) < self.get_parameter('min_track_length').value or
                track.get('stable_frames', 0) < self.get_parameter('min_confidence_frames').value):
                continue

            # Calculate confidence level
            confidence = min(1.0, track['stable_frames'] / 5.0)
            
            # Obstacle marker (cube)
            marker = Marker()
            marker.header = header
            marker.header.frame_id = self.get_parameter('base_frame').value
            marker.ns = "obstacles"
            marker.id = track_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = Point(
                x=float(track['current_position'][0]),
                y=float(track['current_position'][1]),
                z=float(track['current_position'][2]))
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(max(track['size'][0], 0.3))
            marker.scale.y = float(max(track['size'][1], 0.3))
            marker.scale.z = float(max(track['size'][2], 0.3))
            
            # Set color based on obstacle type
            if track['is_dynamic']:
                # Dynamic obstacle - red with confidence-based transparency
                marker.color = ColorRGBA(
                    r=1.0,
                    g=0.0,
                    b=0.0,
                    a=0.5 + 0.3 * confidence)
            else:
                # Static obstacle - blue with confidence-based transparency
                marker.color = ColorRGBA(
                    r=0.0,
                    g=0.0,
                    b=1.0,
                    a=0.3 + 0.2 * confidence)
            
            marker.lifetime.nanosec = int(0.5 * 1e9)
            marker_array.markers.append(marker)

            # Velocity vector (only for dynamic obstacles)
            if track['is_dynamic'] and track['stable_frames'] >= 3 and track['avg_speed'] >= 0.5:
                vel_marker = Marker()
                vel_marker.header = header
                vel_marker.ns = "velocity"
                vel_marker.id = track_id + 10000
                vel_marker.type = Marker.ARROW
                vel_marker.action = Marker.ADD
                
                start = Point(
                    x=float(track['current_position'][0]),
                    y=float(track['current_position'][1]),
                    z=float(track['current_position'][2]))
                
                vel_dir = track['current_velocity'] / (np.linalg.norm(track['current_velocity']) + 1e-6)
                end = Point(
                    x=start.x + vel_dir[0] * 0.5,
                    y=start.y + vel_dir[1] * 0.5,
                    z=start.z + vel_dir[2] * 0.2)
                
                vel_marker.points = [start, end]
                vel_marker.scale.x = 0.1
                vel_marker.scale.y = 0.2
                vel_marker.color = ColorRGBA(
                    r=0.0,
                    g=1.0,
                    b=0.0,
                    a=0.7 + 0.2 * confidence)
                vel_marker.lifetime.nanosec = int(0.5 * 1e9)
                marker_array.markers.append(vel_marker)

        self.marker_pub.publish(marker_array)
def main(args=None):
    rclpy.init(args=args)
    node = AccurateDynamicObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()