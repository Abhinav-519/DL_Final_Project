import torch
from video_processor import process_video

# Example usage
if __name__ == "__main__":
    # We will be loading trained models
    vehicle_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                 path='Weights/best_bdd100k_final.pt')
    sign_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                               path='Weights/best_traffic_signs.pt')
    
    # We will be processing video
    process_video('TestData/ROW_TEST.mp4', vehicle_model, sign_model, 'output_video1.mp4')
