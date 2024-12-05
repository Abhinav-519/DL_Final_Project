from typing import Dict, List, Tuple, Set
import time
from vehicle_types import (
    Vehicle, EgoVehicle, VehicleType, Direction,
    MovementState, TrafficSignal, PRIORITY_VEHICLES
)


class IntersectionManager:
    def __init__(self, reference_line_y: float = 100, frame_width: int = 640):
        self.vehicles: Dict[int, Vehicle] = {}
        self.reference_line_y = reference_line_y
        self.frame_width = frame_width
        self.waiting_queue = []
        self.traffic_signal_state = None
        self.active_signs: Set[str] = set()
        self.ego_vehicle = EgoVehicle(reference_line_y)
        self.start_time = time.time()
        self.last_ego_positions = []
        self.ego_stopped_time = None

    def _update_ego_state(self) -> None:
        current_pos = self.ego_vehicle.bbox[1]
        self.last_ego_positions.append(current_pos)

        # Maintain a rolling window of positions
        if len(self.last_ego_positions) > 5:
            self.last_ego_positions.pop(0)

        if len(self.last_ego_positions) >= 3:
            avg_movement = sum(
                abs(self.last_ego_positions[i] - self.last_ego_positions[i - 1])
                for i in range(1, len(self.last_ego_positions))
            ) / len(self.last_ego_positions)

            if avg_movement < 2.0:
                if self.ego_vehicle.movement_state != MovementState.STOPPED:
                    self.ego_stopped_time = time.time()
                self.ego_vehicle.movement_state = MovementState.STOPPED
            else:
                self.ego_vehicle.movement_state = MovementState.APPROACHING
                self.ego_stopped_time = None

    def _update_vehicle(self, detection: dict, current_time: float) -> None:
        vehicle_id = detection['id']
        current_position = (detection['x'], detection['y'])
        bbox = detection['bbox']
        vehicle_type = VehicleType(detection['class'])
        is_priority = vehicle_type in PRIORITY_VEHICLES
        is_moving = self._is_vehicle_moving(vehicle_id, current_position)

        if vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            vehicle.position = current_position
            vehicle.bbox = bbox
            vehicle.movement_state = (
                MovementState.APPROACHING if is_moving else MovementState.STOPPED
            )
            if not vehicle.has_reached_intersection and self._is_at_intersection(bbox):
                vehicle.has_reached_intersection = True
                vehicle.arrival_time = current_time
                if vehicle_id not in self.waiting_queue:
                    self.waiting_queue.append(vehicle_id)
        else:
            self.vehicles[vehicle_id] = Vehicle(
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
            if self.vehicles[vehicle_id].has_reached_intersection:
                self.waiting_queue.append(vehicle_id)

    def _is_vehicle_moving(self, vehicle_id: int, current_position: Tuple[float, float]) -> bool:
        if vehicle_id in self.vehicles:
            prev_pos = self.vehicles[vehicle_id].position
            return abs(current_position[1] - prev_pos[1]) > 2.0
        return True

    def _is_at_intersection(self, vehicle_bbox: List[int]) -> bool:
        return vehicle_bbox[1] <= self.reference_line_y <= vehicle_bbox[3]

    def _determine_direction(self, position: Tuple[float, float]) -> Direction:
        return Direction.NORTH if position[1] > self.reference_line_y else Direction.SOUTH

    def update_vehicle_positions(self, detections: List[dict]) -> str:
        current_time = time.time()
        active_vehicle_ids = set()

        self._update_ego_state()
        if not self.ego_vehicle.has_reached_intersection and self._is_at_intersection(self.ego_vehicle.bbox):
            self.ego_vehicle.has_reached_intersection = True
            self.ego_vehicle.arrival_time = current_time

        for detection in detections:
            self._update_vehicle(detection, current_time)
            active_vehicle_ids.add(detection['id'])

        self._cleanup_inactive_vehicles(active_vehicle_ids)
        return self._generate_movement_advice()

    def _cleanup_inactive_vehicles(self, active_vehicle_ids: Set[int]) -> None:
        for vehicle_id in list(self.vehicles.keys()):
            if vehicle_id not in active_vehicle_ids:
                del self.vehicles[vehicle_id]
                if vehicle_id in self.waiting_queue:
                    self.waiting_queue.remove(vehicle_id)

    def _generate_movement_advice(self) -> str:
        if self._should_ego_stop():
            return f"Please stop - {self.traffic_signal_state or 'Stop sign'}"

        vehicles_at_intersection = [
            v for v in self.vehicles.values() if self._is_at_intersection(v.bbox)
        ]
        if not vehicles_at_intersection:
            return "You may proceed - No other vehicles at intersection"

        priority_vehicles = [v for v in vehicles_at_intersection if v.is_priority]
        if priority_vehicles:
            first_priority = min(priority_vehicles, key=lambda v: v.arrival_time or float('inf'))
            return f"Please wait - Priority Vehicle {first_priority.id} has right of way"

        return self._determine_right_of_way(vehicles_at_intersection)

    def _should_ego_stop(self) -> bool:
        return self.traffic_signal_state in {TrafficSignal.RED, TrafficSignal.YELLOW} or 'stop' in self.active_signs

    def _determine_right_of_way(self, vehicles: List[Vehicle]) -> str:
        vehicles_before = [v for v in vehicles if (v.bbox[1] + v.bbox[3]) / 2 < self.reference_line_y]
        if vehicles_before:
            first = min(vehicles_before, key=lambda v: v.arrival_time or float('inf'))
            return f"Please wait - Vehicle {first.id} has right of way"

        return "You may proceed - You have right of way"

    def update_traffic_signs(self, sign_detections: List[dict]) -> None:
        self.active_signs.clear()
        for detection in sign_detections:
            sign_class = detection['class']
            self.active_signs.add(sign_class)
            if sign_class in [signal.value for signal in TrafficSignal]:
                self.traffic_signal_state = TrafficSignal(sign_class)

    def get_intersection_state(self) -> dict:
        return {
            'vehicles': {
                vid: {
                    'position': v.position,
                    'movement_state': v.movement_state,
                    'arrival_time': v.arrival_time,
                    'bbox': v.bbox,
                    'is_priority': v.is_priority,
                    'type': v.type.name
                }
                for vid, v in self.vehicles.items()
            },
            'waiting_queue': self.waiting_queue,
            'traffic_signal': self.traffic_signal_state,
            'active_signs': list(self.active_signs)
        }
