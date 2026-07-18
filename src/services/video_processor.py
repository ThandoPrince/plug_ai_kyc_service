import cv2
import os

class VideoProcessor:

    @staticmethod
    def extract_frames(video_path, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        frame_paths = []

        frame_number = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            if frame_number % 10 == 0:
                filename = os.path.join(
                    output_folder,
                    f"frame_{frame_number}.jpg"
                )

                cv2.imwrite(filename, frame)

                frame_paths.append(filename)

            frame_number += 1

        cap.release()

        return frame_paths
    
 