# DL_Final_Project

Project Proposal

Autonomous Vehicle Right-of-Way Decision System for Four-Way Stop Intersections

Introduction

Navigating intersections with stop signs is a fundamental challenge for autonomous vehicles (AVs). It involves determining the right-of-way based on the arrival sequence and traffic rules. Four-way stop intersections require AVs to coordinate effectively with other vehicles—including human-driven ones—while prioritizing safety, rule adherence, and smooth traffic flow.

This project proposes developing a decision-making model that enables AVs to handle right-of-way at four-way stop intersections. By implementing deep learning techniques for traffic sign detection, vehicle tracking, and rule-based decision-making, this system aims to improve AV functionality in urban settings, facilitating safe and efficient intersection navigation.

Objectives

Stop Sign Detection: Develop a model capable of reliably detecting stop signs in varied conditions (e.g., different lighting and weather scenarios).
Vehicle Detection and Tracking: Create a system to detect and track nearby vehicles at intersections, monitoring their arrival sequence and position.
Right-of-Way Decision-Making: Implement a decision-making algorithm that assigns the correct right-of-way based on arrival times, positioning, and traffic rules.
Simulation and Testing: Test and validate the model’s performance in simulated four-way stop scenarios using appropriate alternatives to CARLA.

Methodology

1. Dataset Collection and Preprocessing
LISA Traffic Sign Dataset 
Link: https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset
A dataset containing annotated images of various traffic signs, including stop signs, is suitable for training a detection model.
Berkeley DeepDrive (BDD100K) 
 Link: https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k
A large-scale dataset with labeled images of vehicles and traffic signs under diverse driving conditions.
Synthetic Data Generation: Use alternative simulation tools like LGSVL or Webots to generate labeled intersection scenarios for additional model training and testing.

2. Model Design and Training
 Stop Sign Detection: Utilize a pre-trained model such as YOLOv5 or Faster R-CNN, fine-tuned on the LISA and BDD100K datasets. Data augmentation (e.g., brightness adjustment, scaling) will be applied to improve robustness under various conditions.
Vehicle Detection and Tracking: Implement YOLO for real-time vehicle detection, combined with Deep SORT tracking, to continuously monitor other vehicles at intersections.
Right-of-Way Decision-Making: Develop a decision-making module using a rule-based approach or reinforcement learning techniques. This module will prioritize vehicles based on arrival order and positioning while simulating human-like driving decisions.

3. Simulation and Testing
   Simulation Tools: 
Conduct tests using LGSVL or Webots to model intersection scenarios and evaluate the AV’s decision-making performance in real-time. Include diverse scenarios to simulate different arrival orders, unexpected stops, and close-call situations.
Evaluation Metrics:
Detection Accuracy: Assess the system’s success in recognizing stop signs and tracking vehicles under varied conditions.
Right-of-Way Accuracy: Measure the accuracy of right-of-way decisions, ensuring compliance with traffic rules and avoidance of conflicts.
Safety and Coordination: Evaluate safety by measuring the model’s response time, decision confidence, and ability to handle unexpected behaviors from human drivers.

Key Challenges and Solutions

1. Complexity of Right-of-Way Decisions: Intersections with multiple vehicles require precise timing and contextual awareness. To address this, the project will use a combination of rule-based decision-making and reinforcement learning, enabling adaptive responses based on real-time data.

2. Environmental Variability: Variations in lighting and weather can affect stop sign detection. Training on a broad dataset, augmented with synthetic images from simulation environments, will increase robustness.

3. Predicting Human Behavior: Human drivers may act unpredictably. A predictive behavior model will be developed to infer likely actions based on observed vehicle positions, supporting timely and adaptive AV responses.

Expected Outcomes

By the end of this project, the AV right-of-way decision system will:
Accurately detect stop signs and identify vehicles across varied environmental conditions.
Make correct, rule-compliant decisions for yielding or proceeding based on traffic protocols.
Enhance the safety and coordination of AVs with human drivers, supporting reliable and smooth interactions at intersections.

References
"Rule-Based Decision-Making System for Autonomous Vehicles at Intersections with Mixed Traffic Environment," IEEE Xplore. 
This paper provides insights into decision-making frameworks for AVs in complex traffic environments.
  	Link: https://ieeexplore.ieee.org/document/9565085
Jialin Li et al., "A Shared-Road-Rights Driving Strategy for Right-of-Way Conflicts," explores shared road strategies for AVs, emphasizing conflict-free, cooperative decision-making at intersections.

This project will deliver a right-of-way system that integrates detection, tracking, and adaptive decision-making, advancing AV technology for safer, more efficient urban navigation.

