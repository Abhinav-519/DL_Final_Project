# We will be importing required dependencies from standard library
from typing import Dict, List, Tuple, Set
import time

# We will be importing our custom vehicle types and enums from vehicle_types.py
from vehicle_types import (
    Vehicle, EgoVehicle, VehicleType, Direction,
    MovementState, TrafficSignal, PRIORITY_VEHICLES
)

# We will be managing intersection traffic and vehicle control logic by implementing a system
# that tracks vehicle positions, determines right of way, and provides movement advice
class IntersectionManager:
    def __init__(self, reference_line_y: float = 100, frame_width: int = 640):
        # We will be initializing core tracking dictionaries and lists
        self.vehicles: Dict[int, Vehicle] = {}  # Maps vehicle IDs to their objects
        self.reference_line_y = reference_line_y  # Y-coordinate for intersection line
        self.frame_width = frame_width  # Width of camera frame
        self.waiting_queue = []  # Order of vehicles waiting at intersection
        
        # We will be tracking traffic control states
        self.current_right_of_way = None  # Vehicle with current right of way
        self.traffic_signal_state = None  # Current traffic signal state
        self.active_signs: Set[str] = set()  # Active traffic signs
        
        # We will be managing ego vehicle state
        self.ego_vehicle = EgoVehicle(reference_line_y)  # Our vehicle with dashcam
        self.start_time = time.time()  # System start time
        self.last_ego_positions = []  # Track last 5 positions for movement detection
        self.ego_stopped_time = None  # Timestamp when ego vehicle stopped

    # We will be updating ego vehicle's state by analyzing its recent position history
    def _update_ego_state(self) -> None:
        current_pos = self.ego_vehicle.bbox[1]  # Get Y position from bbox
        self.last_ego_positions.append(current_pos)
        
        # We will be maintaining a rolling window of 5 positions
        if len(self.last_ego_positions) > 5:
            self.last_ego_positions.pop(0)  # Remove oldest position
            
        # We will be calculating movement only if we have enough position history
        if len(self.last_ego_positions) >= 3:
            # We will be measuring position changes
            movements = [abs(self.last_ego_positions[i] - self.last_ego_positions[i-1]) 
                        for i in range(1, len(self.last_ego_positions))]
            avg_movement = sum(movements) / len(movements)
            
            # We will be using 2.0 pixels as threshold for stopped state
            if avg_movement < 2.0:  
                if self.ego_vehicle.movement_state != MovementState.STOPPED:
                    self.ego_stopped_time = time.time()  # Record stop time
                self.ego_vehicle.movement_state = MovementState.STOPPED
            else:
                self.ego_vehicle.movement_state = MovementState.APPROACHING
                self.ego_stopped_time = None

    # We will be updating positions of all vehicles based on new detections
    def update_vehicle_positions(self, detections: List[dict]) -> str:
        current_time = time.time()
        current_vehicles = set()  # Track active vehicles this frame
        
        # We will be updating ego vehicle first
        self._update_ego_state()
        
        # We will be checking if ego vehicle just reached intersection
        if (not self.ego_vehicle.has_reached_intersection and 
            self._is_at_intersection(self.ego_vehicle.bbox)):
            self.ego_vehicle.arrival_time = current_time
            self.ego_vehicle.has_reached_intersection = True
        
        # We will be processing each detected vehicle
        for detection in detections:
            vehicle_id = detection['id']
            current_vehicles.add(vehicle_id)  # Mark vehicle as active
            current_position = (detection['x'], detection['y'])
            bbox = detection['bbox']
            
            # We will be determining if vehicle is moving
            is_moving = True
            if vehicle_id in self.vehicles:  # Check previous position if available
                prev_pos = self.vehicles[vehicle_id].position
                movement = abs(current_position[1] - prev_pos[1])
                is_moving = movement > 2.0  # Same threshold as ego vehicle
            
            vehicle_type = VehicleType(detection['class'])
            is_priority = vehicle_type in PRIORITY_VEHICLES
            
            # We will be updating existing vehicles or creating new ones
            if vehicle_id in self.vehicles:
                # Update existing vehicle
                vehicle = self.vehicles[vehicle_id]
                vehicle.position = current_position
                vehicle.bbox = bbox
                vehicle.is_priority = is_priority
                vehicle.movement_state = (MovementState.APPROACHING if is_moving 
                                        else MovementState.STOPPED)
                
                # We will be checking if vehicle newly arrived at intersection
                if self._is_at_intersection(bbox) and not vehicle.has_reached_intersection:
                    vehicle.arrival_time = current_time
                    vehicle.has_reached_intersection = True
                    if vehicle_id not in self.waiting_queue:
                        self.waiting_queue.append(vehicle_id)
            else:
                # We will be creating new vehicle
                vehicle = Vehicle(
                    id=vehicle_id,
                    type=vehicle_type,
                    position=current_position,
                    bbox=bbox,
                    direction=self._determine_direction(current_position),
                    arrival_time=current_time if self._is_at_intersection(bbox) else None,
                    movement_state=MovementState.APPROACHING,
                    is_priority=is_priority,
                    has_reached_intersection=self._is_at_intersection(bbox)
                )
                self.vehicles[vehicle_id] = vehicle
                if vehicle.has_reached_intersection:
                    self.waiting_queue.append(vehicle_id)
        
        # We will be cleaning up vehicles that disappeared
        for vehicle_id in list(self.vehicles.keys()):
            if vehicle_id not in current_vehicles:
                self.remove_vehicle(vehicle_id)
                
        return self._generate_movement_advice()

    # We will be checking if vehicle has reached the intersection
    def _is_at_intersection(self, vehicle_bbox: List[int]) -> bool:
        bbox_top = vehicle_bbox[1]  # Y coordinate of top edge
        bbox_bottom = vehicle_bbox[3]  # Y coordinate of bottom edge
        return bbox_top <= self.reference_line_y <= bbox_bottom

    # We will be determining if ego vehicle should stop
    def _should_ego_stop(self) -> bool:
        if self.traffic_signal_state == TrafficSignal.RED:
            return True  # Stop on red light
        if self.traffic_signal_state == TrafficSignal.YELLOW:
            return True  # Stop on yellow light
        if 'stop' in self.active_signs:
            return True  # Stop on stop sign
        return False

    # We will be generating appropriate movement advice based on intersection state
    def _generate_movement_advice(self) -> str:
        # We will be checking traffic signals first
        if self._should_ego_stop():
            return "Please stop" + (
                " - Red light" if self.traffic_signal_state == TrafficSignal.RED else
                " - Yellow light" if self.traffic_signal_state == TrafficSignal.YELLOW else
                " - Stop sign" if 'stop' in self.active_signs else ""
            )
        
        # We will be getting list of vehicles at intersection
        vehicles_at_intersection = [v for v in self.vehicles.values() 
                                  if self._is_at_intersection(v.bbox)]
        
        if not vehicles_at_intersection:
            return "You may proceed - No other vehicles at intersection"
        
        # We will be checking priority vehicles (emergency vehicles) first
        priority_vehicles = [v for v in vehicles_at_intersection if v.is_priority]
        if priority_vehicles:
            first_priority = min(priority_vehicles, key=lambda x: x.arrival_time or float('inf'))
            return f"Please wait - Priority Vehicle {first_priority.id} has right of way"
        
        # We will be grouping vehicles by their relative position
        vehicles_before_us = []  # Vehicles ahead of us
        vehicles_with_us = []    # Vehicles at similar position
        vehicles_after_us = []   # Vehicles behind us
        
        for vehicle in vehicles_at_intersection:
            vehicle_center_y = (vehicle.bbox[1] + vehicle.bbox[3]) / 2
            distance_to_reference = vehicle_center_y - self.reference_line_y
            
            # We will be using 20 pixel threshold for position grouping
            if distance_to_reference < -20:
                vehicles_before_us.append(vehicle)
            elif distance_to_reference > 20:
                vehicles_after_us.append(vehicle)
            else:
                vehicles_with_us.append(vehicle)
        
        # We will be checking stopped vehicles
        stopped_vehicles = [v for v in vehicles_at_intersection 
                           if v.movement_state == MovementState.STOPPED]
        
        # We will be handling case when we're moving but others are stopped
        if (self.ego_vehicle.movement_state == MovementState.APPROACHING and 
            stopped_vehicles):
            first_stopped = min(stopped_vehicles, 
                              key=lambda x: x.arrival_time or float('inf'))
            others = [v.id for v in vehicles_at_intersection if v != first_stopped]
            wait_msg = (f" (Vehicle{'s' if len(others)>1 else ''} "
                       f"{', '.join(map(str, others))} also waiting)" if others else "")
            return f"Please wait - Vehicle {first_stopped.id} stopped first{wait_msg}"
        
        # We will be handling case when we're stopped
        if self.ego_vehicle.movement_state == MovementState.STOPPED:
            moving_vehicles = [v for v in vehicles_at_intersection 
                             if v.movement_state == MovementState.APPROACHING]
            if not moving_vehicles:
                if vehicles_with_us:
                    # We will be checking right-of-way for vehicles at similar positions
                    vehicles_to_right = [v for v in vehicles_with_us 
                                       if v.position[0] > self.frame_width/2]
                    if not vehicles_to_right:
                        others = [v.id for v in vehicles_with_us]
                        wait_msg = (f" (Vehicle{'s' if len(others)>1 else ''} "
                                  f"{', '.join(map(str, others))} should wait)" if others else "")
                        return f"You may proceed - You're rightmost{wait_msg}"
                    else:
                        rightmost = min(vehicles_to_right, key=lambda x: x.position[0])
                        return f"Please wait - Vehicle {rightmost.id} has right of way (rightmost position)"
                return "You may proceed - You stopped first"
        
        # We will be handling position-based priority
        if vehicles_before_us:
            first_vehicle = min(vehicles_before_us, key=lambda x: x.position[1])
            others = [v.id for v in vehicles_at_intersection if v != first_vehicle]
            wait_msg = (f" (Vehicle{'s' if len(others)>1 else ''} "
                       f"{', '.join(map(str, others))} also waiting)" if others else "")
            return f"Please wait - Vehicle {first_vehicle.id} is ahead{wait_msg}"
        
        if vehicles_with_us:
            vehicles_to_right = [v for v in vehicles_with_us 
                               if v.position[0] > self.frame_width/2]
            if not vehicles_to_right:
                others = [v.id for v in vehicles_with_us]
                wait_msg = (f" (Vehicle{'s' if len(others)>1 else ''} "
                           f"{', '.join(map(str, others))} should wait)" if others else "")
                return f"You may proceed - You have right of way{wait_msg}"
            else:
                rightmost = min(vehicles_to_right, key=lambda x: x.position[0])
                return f"Please wait - Vehicle {rightmost.id} has right of way (rightmost position)"
        
        if vehicles_after_us:
            others = [v.id for v in vehicles_after_us]
            wait_msg = (f" (Vehicle{'s' if len(others)>1 else ''} "
                       f"{', '.join(map(str, others))} behind you)" if others else "")
            return f"You may proceed{wait_msg}"
        
        return "You may proceed with caution"

    # We will be updating traffic sign states from new detections
    def update_traffic_signs(self, sign_detections: List[dict]):
        self.active_signs.clear()  # Reset active signs
        for detection in sign_detections:
            sign_class = detection['class']
            self.active_signs.add(sign_class)
            # We will be updating traffic signal state if sign is a signal
            if sign_class in [signal.value for signal in TrafficSignal]:
                self.traffic_signal_state = TrafficSignal(sign_class)
    
    # We will be determining vehicle direction based on its position
    def _determine_direction(self, position: Tuple[float, float]) -> Direction:
        if position[1] > self.reference_line_y:
            return Direction.NORTH  # Moving towards intersection
        return Direction.SOUTH  # Moving away from intersection
    
    # We will be removing vehicles from tracking system
    def remove_vehicle(self, vehicle_id: int):
        if vehicle_id in self.vehicles:
            del self.vehicles[vehicle_id]
        if vehicle_id in self.waiting_queue:
            self.waiting_queue.remove(vehicle_id)

    # We will be providing current state of intersection for external use
    def get_intersection_state(self) -> dict:
        return {
            'vehicles': {vid: {
                'position': v.position,
                'movement_state': v.movement_state,
                'arrival_time': v.arrival_time,
                'bbox': v.bbox,
                'is_priority': v.is_priority,
                'type': v.type.name
            } for vid, v in self.vehicles.items()},
            'waiting_queue': self.waiting_queue,
            'traffic_signal': self.traffic_signal_state,
            'current_right_of_way': self.current_right_of_way,
            'active_signs': list(self.active_signs)
        }
