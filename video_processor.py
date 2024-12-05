import cv2
import torch
import time
from typing import List

# We will be importing our custom types and manager
from vehicle_types import VehicleType, PRIORITY_VEHICLES
from intersection_manager import IntersectionManager

# We will be implementing video processing functionality for intersection management
def process_video(
    video_path: str, 
    vehicle_model: torch.nn.Module,
    sign_model: torch.nn.Module,
    output_path: str = "output_video.mp4"
):
    # We will be initializing video capture and checking for errors
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
        
    # We will be getting video properties for output configuration
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # We will be setting up video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # We will be initializing intersection manager with reference line at 60% of frame height
    reference_line_y = int(height * 0.6)
    intersection = IntersectionManager(reference_line_y=reference_line_y, frame_width=width)
    
    # We will be setting detection confidence threshold
    conf_threshold = 0.5
    frame_counter = 0
    
    try:
        # We will be processing each frame in the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_counter += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # We will be getting detections from both models
            vehicle_results = vehicle_model(rgb_frame)
            sign_results = sign_model(rgb_frame)
            
            # We will be processing vehicle detections
            vehicle_detections = []
            for i, (*xyxy, conf, cls) in enumerate(vehicle_results.xyxy[0]):
                if conf > conf_threshold and int(cls) <= 7:  # Only vehicle classes
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    vehicle_detections.append({
                        'id': i,
                        'class': int(cls),
                        'x': float(x_center),
                        'y': float(y_center),
                        'bbox': [int(x) for x in xyxy]
                    })
            
            # We will be processing traffic sign detections
            sign_detections = []
            for *xyxy, conf, cls in sign_results.xyxy[0]:
                if conf > conf_threshold:
                    class_name = sign_results.names[int(cls)]
                    sign_detections.append({
                        'class': class_name,
                        'bbox': [int(x) for x in xyxy]
                    })
            
            # We will be updating intersection state and getting movement advice
            intersection.update_traffic_signs(sign_detections)
            advice = intersection.update_vehicle_positions(vehicle_detections)
            
            # We will be drawing reference line
            cv2.line(frame, (0, reference_line_y), (width, reference_line_y), 
                    (0, 255, 0), 2)
            
            # We will be drawing ego vehicle indicator
            ego = intersection.ego_vehicle
            ego_color = (0, 255, 255)  # Yellow for ego vehicle
            if ego.has_reached_intersection:
                cv2.rectangle(frame, 
                            (ego.bbox[0], ego.bbox[1]),
                            (ego.bbox[2], ego.bbox[3]),
                            ego_color, 2)
                ego_label = "OUR VEHICLE"
                if ego.arrival_time:
                    arrival_time = time.strftime('%H:%M:%S', time.localtime(ego.arrival_time))
                    ego_label += f" | Arrived: {arrival_time}"
                
                label_size = cv2.getTextSize(ego_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, 
                            (ego.bbox[0], ego.bbox[1]-label_size[1]-10),
                            (ego.bbox[0] + label_size[0], ego.bbox[1]),
                            (0, 0, 0),
                            -1)
                cv2.putText(frame, ego_label, 
                          (ego.bbox[0], ego.bbox[1]-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, ego_color, 2)
            
            # We will be drawing other vehicles with their status
            for det in vehicle_detections:
                x1, y1, x2, y2 = det['bbox']
                vehicle_type = VehicleType(det['class'])
                is_priority = vehicle_type in PRIORITY_VEHICLES
                is_at_intersection = intersection._is_at_intersection(det['bbox'])
                
                # We will be color coding vehicles based on their status
                if is_priority:
                    color = (0, 0, 255)  # Red for priority vehicles
                else:
                    color = (255, 255, 0) if is_at_intersection else (255, 0, 0)
                
                # We will be drawing bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # We will be preparing vehicle label with status
                vehicle_info = []
                vehicle_info.append(f"{vehicle_type.name}#{det['id']}")
                if is_priority:
                    vehicle_info.append("PRIORITY")
                if is_at_intersection:
                    vehicle_info.append("AT INTERSECTION")
                    if det['id'] in intersection.vehicles:
                        v = intersection.vehicles[det['id']]
                        if v.arrival_time:
                            arrival_time = time.strftime('%H:%M:%S', time.localtime(v.arrival_time))
                            vehicle_info.append(f"Arrived: {arrival_time}")
                
                # We will be drawing vehicle label
                label = " | ".join(vehicle_info)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, 
                            (x1, y1-label_size[1]-10),
                            (x1 + label_size[0], y1),
                            (0, 0, 0),
                            -1)
                cv2.putText(frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, color, 2)
            
            # We will be drawing traffic signs
            for det in sign_detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, det['class'], 
                          (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (0, 255, 0), 2)
            
            # We will be drawing movement advice with styling
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            
            # We will be handling multi-part advice messages
            advice_parts = advice.split('(')
            main_advice = advice_parts[0].strip()
            secondary_advice = f"({advice_parts[1]}" if len(advice_parts) > 1 else ""
            
            # We will be drawing main advice
            text_size = cv2.getTextSize(main_advice, font, font_scale, font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 60
            
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0),
                         -1)
            cv2.putText(frame, main_advice, (text_x, text_y), font, font_scale, 
                       (255, 255, 255), font_thickness)
            
            # We will be drawing secondary advice if it exists
            if secondary_advice:
                text_size = cv2.getTextSize(secondary_advice, font, 0.7, 1)[0]
                text_x = (width - text_size[0]) // 2
                text_y = height - 30
                
                cv2.rectangle(frame, 
                             (text_x - 10, text_y - text_size[1] - 10),
                             (text_x + text_size[0] + 10, text_y + 10),
                             (0, 0, 0),
                             -1)
                cv2.putText(frame, secondary_advice, (text_x, text_y), font, 0.7, 
                           (255, 255, 255), 1)
            
            # We will be writing the processed frame to output video
            out.write(frame)
                
    finally:
        # We will be cleaning up resources
        cap.release()
        out.release()
        print("Processing Completed!")
