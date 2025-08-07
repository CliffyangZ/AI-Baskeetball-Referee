import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import deque
from scipy.signal import find_peaks, savgol_filter
import time
import tempfile
from gtts import gTTS
from playsound import playsound
import sys

sys.path.append("/Users/cliffyang/Documents/Program/Basketball_ai_project/models")
sys.path.append("/Users/cliffyang/Documents/Program/Basketball_ai_project/data")

class StepCounter:
    """
    基於 YOLO 姿態估計的步數計算器
    """
    
    def __init__(self, model_path='models/pt_models/yolov8s-pose.pt'):
        """
        初始化步數計算器
        
        Args:
            model_path: YOLO 姿態估計模型路徑
        """
        self.model = YOLO(model_path)
        
        # COCO 姿態關鍵點索引
        self.keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        # 步數計算參數
        self.step_count = 0
        self.last_step_time = 0
        self.min_step_interval = 0.3  # 最小步伐間隔（秒）
        
        # 數據緩衝區
        self.buffer_size = 30  # 保存30幀的數據
        self.ankle_positions = {
            'left': deque(maxlen=self.buffer_size),
            'right': deque(maxlen=self.buffer_size)
        }
        self.knee_angles = {
            'left': deque(maxlen=self.buffer_size),
            'right': deque(maxlen=self.buffer_size)
        }
        self.hip_heights = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        
        # 步態分析參數
        self.walking_threshold = 10  # 腳踝移動閾值（像素）
        self.angle_threshold = 15    # 膝蓋角度變化閾值（度）
        self.confidence_threshold = 0.5  # 關鍵點信心度閾值
        
        # 狀態追蹤
        self.last_peak_times = {'left': 0, 'right': 0}
        self.gait_cycle_data = []
        
        # 生成音效
        try:
            self.tts = gTTS(text="Step", lang="en")
            self.temp_file = tempfile.NamedTemporaryFile(delete=False)
            self.tts.save(self.temp_file.name)
            self.sound_enabled = True
        except Exception as e:
            print(f"Warning: Could not initialize audio: {e}")
            self.sound_enabled = False
    
    def calculate_distance(self, p1, p2):
        """計算兩點間距離"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_angle(self, p1, p2, p3):
        """
        計算三點形成的角度
        p2 為頂點
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # 防止數值錯誤
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def extract_keypoints(self, results):
        """
        從 YOLO 結果中提取關鍵點
        """
        keypoints_data = []
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.data  # shape: [N, 17, 3] (x, y, confidence)
                
                for person_kpts in keypoints:
                    person_data = {}
                    for name, idx in self.keypoint_indices.items():
                        x, y, conf = person_kpts[idx]
                        if conf > self.confidence_threshold:
                            person_data[name] = (float(x), float(y), float(conf))
                        else:
                            person_data[name] = None
                    
                    keypoints_data.append(person_data)
        
        return keypoints_data

    def analyze_gait_cycle(self, keypoints):
        """
        分析步態週期
        """
        current_time = time.time()
        
        # 提取關鍵點
        left_ankle = keypoints.get('left_ankle')
        right_ankle = keypoints.get('right_ankle')
        left_knee = keypoints.get('left_knee')
        right_knee = keypoints.get('right_knee')
        left_hip = keypoints.get('left_hip')
        right_hip = keypoints.get('right_hip')
        
        if not all([left_ankle, right_ankle, left_knee, right_knee, left_hip, right_hip]):
            return False
        
        # 計算腳踝位置
        left_ankle_pos = (left_ankle[0], left_ankle[1])
        right_ankle_pos = (right_ankle[0], right_ankle[1])
        
        # 計算膝蓋角度
        left_knee_angle = self.calculate_angle(left_hip[:2], left_knee[:2], left_ankle[:2])
        right_knee_angle = self.calculate_angle(right_hip[:2], right_knee[:2], right_ankle[:2])
        
        # 計算髖部高度（用於檢測上下運動）
        hip_height = (left_hip[1] + right_hip[1]) / 2
        
        # 更新緩衝區
        self.ankle_positions['left'].append(left_ankle_pos)
        self.ankle_positions['right'].append(right_ankle_pos)
        self.knee_angles['left'].append(left_knee_angle)
        self.knee_angles['right'].append(right_knee_angle)
        self.hip_heights.append(hip_height)
        self.timestamps.append(current_time)
        
        return True
    
    def detect_steps_ankle_method(self):
        """
        基於腳踝垂直運動的步數檢測
        """
        if len(self.ankle_positions['left']) < 10:
            return 0
        
        steps_detected = 0
        current_time = time.time()
        
        # 分析左右腳的垂直運動
        for side in ['left', 'right']:
            ankle_y = [pos[1] for pos in self.ankle_positions[side]]
            
            if len(ankle_y) < 10:
                continue
            
            # 平滑化數據
            smoothed_y = savgol_filter(ankle_y, window_length=5, polyorder=2)
            
            # 尋找峰值（腳抬起的最高點）
            peaks, properties = find_peaks(
                -np.array(smoothed_y),  # 負值以找到最低點
                height=None,
                distance=5,  # 最小峰值距離
                prominence=self.walking_threshold
            )
            
            # 檢查是否有新的步伐
            if len(peaks) > 0:
                last_peak_idx = peaks[-1]
                last_peak_time = self.timestamps[last_peak_idx] if last_peak_idx < len(self.timestamps) else current_time
                
                # 檢查時間間隔
                if (last_peak_time - self.last_peak_times[side] > self.min_step_interval and
                    last_peak_time > self.last_peak_times[side]):
                    
                    steps_detected += 1
                    self.last_peak_times[side] = last_peak_time
        
        return steps_detected
    
    def detect_steps_knee_angle_method(self):
        """
        基於膝蓋角度變化的步數檢測
        """
        if len(self.knee_angles['left']) < 15:
            return 0
        
        steps_detected = 0
        current_time = time.time()
        
        for side in ['left', 'right']:
            angles = list(self.knee_angles[side])
            
            if len(angles) < 15:
                continue
            
            # 計算角度變化率
            angle_changes = np.diff(angles)
            
            # 尋找顯著的角度變化（彎曲到伸直的過程）
            significant_changes = []
            for i in range(len(angle_changes) - 5):
                window = angle_changes[i:i+5]
                if np.sum(window) > self.angle_threshold:  # 角度增加（伸直）
                    significant_changes.append(i)
            
            # 檢測步伐
            if significant_changes:
                last_change_idx = significant_changes[-1]
                last_change_time = self.timestamps[last_change_idx] if last_change_idx < len(self.timestamps) else current_time
                
                if (last_change_time - self.last_peak_times[side] > self.min_step_interval and
                    last_change_time > self.last_peak_times[side]):
                    
                    steps_detected += 1
                    self.last_peak_times[side] = last_change_time
        
        return steps_detected
    
    def detect_steps_combined_method(self):
        """
        結合多種方法的步數檢測
        """
        ankle_steps = self.detect_steps_ankle_method()
        knee_steps = self.detect_steps_knee_angle_method()
        
        # 使用加權平均或投票機制
        combined_steps = max(ankle_steps, knee_steps)  # 取較大值
        
        return combined_steps
    
    def calculate_gait_parameters(self):
        """
        計算步態參數
        """
        if len(self.timestamps) < 10:
            return {}
        
        # 計算步頻（步/分鐘）
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span > 0:
            cadence = (self.step_count / time_span) * 60
        else:
            cadence = 0
        
        # 計算步長估計（基於腳踝移動距離）
        if len(self.ankle_positions['left']) >= 2:
            left_distance = self.calculate_distance(
                self.ankle_positions['left'][-1],
                self.ankle_positions['left'][0]
            )
            right_distance = self.calculate_distance(
                self.ankle_positions['right'][-1],
                self.ankle_positions['right'][0]
            )
            avg_stride = (left_distance + right_distance) / 2
        else:
            avg_stride = 0
        
        return {
            'cadence': cadence,
            'avg_stride': avg_stride,
            'total_steps': self.step_count,
            'walking_time': time_span
        }
    
    def process_frame(self, frame):
        """
        處理單幀圖像
        """
        # YOLO 姿態檢測
        results = self.model(frame, verbose=False, conf=0.5)
        
        # 提取關鍵點
        keypoints_list = self.extract_keypoints(results)
        
        if not keypoints_list:
            return frame, 0
        
        # 使用第一個檢測到的人（可以擴展為多人）
        keypoints = keypoints_list[0]
        
        # 分析步態
        new_steps = 0
        if self.analyze_gait_cycle(keypoints):
            # 檢測新步數
            new_steps = self.detect_steps_combined_method()
            if new_steps > 0:
                self.step_count += new_steps
                # 播放步數聲音 (如果啟用)
                if self.sound_enabled:
                    try:
                        playsound(self.temp_file.name)
                    except Exception as e:
                        print(f"Warning: Could not play sound: {e}")
                        self.sound_enabled = False  # 禁用聲音以避免更多錯誤
        
        # 繪製結果
        annotated_frame = self.draw_pose_and_info(frame, results, keypoints)
        
        return annotated_frame, new_steps
    
    def draw_pose_and_info(self, frame, results, keypoints):
        """
        在圖像上繪製姿態和步數資訊
        """
        # 繪製姿態
        annotated_frame = results[0].plot()
        
        # 添加步數資訊
        info_text = [
            f"Steps: {self.step_count}",
            f"Time: {time.time() - (self.timestamps[0] if self.timestamps else time.time()):.1f}s"
        ]
        
        # 計算步態參數
        gait_params = self.calculate_gait_parameters()
        if gait_params:
            info_text.append(f"Cadence: {gait_params['cadence']:.1f} steps/min")
        
        # 繪製資訊文字
        y_offset = 30
        for text in info_text:
            cv2.putText(annotated_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 40
        
        # 繪製腳踝軌跡
        if len(self.ankle_positions['left']) > 1:
            for i in range(1, len(self.ankle_positions['left'])):
                pt1 = tuple(map(int, self.ankle_positions['left'][i-1]))
                pt2 = tuple(map(int, self.ankle_positions['left'][i]))
                cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 2)  # 左腳藍色
        
        if len(self.ankle_positions['right']) > 1:
            for i in range(1, len(self.ankle_positions['right'])):
                pt1 = tuple(map(int, self.ankle_positions['right'][i-1]))
                pt2 = tuple(map(int, self.ankle_positions['right'][i]))
                cv2.line(annotated_frame, pt1, pt2, (0, 0, 255), 2)  # 右腳紅色
        
        return annotated_frame
    
    def run_video_analysis(self, video_path=0, save_path=None):
        """
        執行視頻步數分析
        """
        cap = cv2.VideoCapture(video_path)
        
        # 視頻屬性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 視頻寫入
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        print("開始步數計算... 按 'q' 退出")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 處理幀
            annotated_frame, new_steps = self.process_frame(frame)
            
            # 顯示結果
            cv2.imshow('Step Counter', annotated_frame)
            
            # 保存視頻
            if save_path:
                out.write(annotated_frame)
            
            # 退出條件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 最終統計
        final_stats = self.calculate_gait_parameters()
        print("\n=== 步數統計 ===")
        print(f"總步數: {self.step_count}")
        print(f"步頻: {final_stats.get('cadence', 0):.1f} 步/分鐘")
        print(f"運動時間: {final_stats.get('walking_time', 0):.1f} 秒")
        
        # 釋放資源
        cap.release()
        if save_path:
            out.release()
        cv2.destroyAllWindows()
        
        return final_stats
    
    def reset_counter(self):
        """重置計數器"""
        self.step_count = 0
        self.ankle_positions['left'].clear()
        self.ankle_positions['right'].clear()
        self.knee_angles['left'].clear()
        self.knee_angles['right'].clear()
        self.hip_heights.clear()
        self.timestamps.clear()
        self.last_peak_times = {'left': 0, 'right': 0}


# 使用範例
if __name__ == "__main__":
    # 初始化步數計算器
    counter = StepCounter('models/pt_models/yolov8s-pose.pt')
    
    # 執行視頻分析
    stats = counter.run_video_analysis("data/video/Travel.mov", None)
