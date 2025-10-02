import cv2
import mediapipe as mp
import pandas as pd

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.fps = None
        self.total_frames = None
        self.landmark_names = [
            'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
            'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
            'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
            'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
            'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
            'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
        ]

    def extract_coordinates(self, output_csv="coordinates_raw.csv"):
        cap = cv2.VideoCapture(self.video_path)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_data = []
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    frame_data = {"frame": frame_idx, "timestamp": frame_idx / self.fps}
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        name = self.landmark_names[idx]
                        frame_data[f"{name}_x"] = landmark.x
                        frame_data[f"{name}_y"] = landmark.y
                        frame_data[f"{name}_z"] = landmark.z
                        frame_data[f"{name}_visibility"] = landmark.visibility
                    all_data.append(frame_data)

                frame_idx += 1
            cap.release()

        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        return df, self.fps, self.total_frames
