# We will be importing required libraries for data structures and type hints
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
from enum import Enum
import time

# We will be defining different types of vehicles that can be detected in the system
class VehicleType(Enum):
    PEDESTRIAN = 0
    RIDER = 1
    CAR = 2
    TRUCK = 3
    BUS = 4
    MOTORCYCLE = 5
    BICYCLE = 6
    CARAVAN = 7
    AMBULANCE = 8
    TRAFFIC_SIGN = 9   
    TRAFFIC_LIGHT = 10 

# We will be specifying possible directions for vehicle movement
class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

# We will be tracking different states of vehicle movement at intersections
class MovementState(Enum):
    APPROACHING = 0
    STOPPED = 1
    PROCEEDING = 2
    CLEARED = 3

# We will be handling different traffic signal states for intersection management
class TrafficSignal(Enum):
    RED = "light_red"
    GREEN = "light_green"
    YELLOW = "light_yellow"

# We will be identifying vehicles that have priority at intersections
PRIORITY_VEHICLES = {
    VehicleType.CARAVAN,
    VehicleType.MOTORCYCLE,
    VehicleType.BUS,
    VehicleType.TRUCK,
    VehicleType.AMBULANCE
}

# We will be storing vehicle information and tracking their states
@dataclass
class Vehicle:
    id: int
    type: VehicleType
    position: Tuple[float, float]
    bbox: List[int]
    direction: Direction
    arrival_time: float
    movement_state: MovementState
    speed: float = 0.0
    is_priority: bool = False
    has_reached_intersection: bool = False

# We will be managing our own vehicle with dashcam capabilities
@dataclass
class EgoVehicle(Vehicle):
    def __init__(self, reference_line_y: float):
        y_pos = reference_line_y + 50
        super().__init__(
            id=-1,
            type=VehicleType.CAR,
            position=(320, y_pos),
            bbox=[270, y_pos-25, 370, y_pos+25],
            direction=Direction.NORTH,
            arrival_time=None,
            movement_state=MovementState.APPROACHING,
            speed=0.0,
            is_priority=False,
            has_reached_intersection=False
        )
