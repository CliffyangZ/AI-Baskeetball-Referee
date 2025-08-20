# Ball Don't Lie: AI Basketball Referee Using ByteTrack Algorithm and YOLOv8


## 專題背景 Project Background
	Basketball is a prominent sport in Taiwan, with leagues played across schools, universities, and communities, and influence spanning even to its national media and film. This fast-paced ball game is played with a sizable set of rules, implemented by an official called a “referee.” Accurate officiating is crucial to ensure the safety of players and the integrity of competition. However, basketball’s high level of physical contact combined with quick and sharp plays make refereeing challenging. Human referees use their personal judgment in officiating, which at times may give controversial calls induced by fatigue, bias, or stress. As a result, these calls often become a disputable matter in amateur games and professional leagues.

Figure 1.1. Behind the scenes for the film “We Are Champions!” Photo by Touch of Light Films.
	The use of technology in aiding sports officiating has been widespread globally. Systems like Sony’s Hawk-Eye in the National Basketball Association (NBA) and the Video Assistant Referee (VAR) in soccer utilize artificial intelligence (AI) to aid in accuracy and fairness in calls and decision-making. Nevertheless, basketball tournaments played in community and school competitions employ little to no technological support. Most basketball games played in Taiwan rely entirely on human referees without access to replay systems or advanced officiating tools.
	This gap presents an opportunity to apply AI and computer vision (CV) into basketball refereeing. A useful CV innovation is the You Only Look Once (YOLO) series of models for object detection, praised for its robustness and efficiency yet compact model size. Object tracking is another aspect of CV that is needed for basketball officiating, which can be provided by the ByteTrack algorithm. This algorithm is praised for its multiple-object detection accuracy which is necessary to properly track player positions, movements, and interactions on the court. The integration of the YOLOv8 model for object detection and ByteTrack algorithm for object tracking allows for the creation of an AI basketball referee capable of real-time monitoring and officiating.

Figure 1.2. ByteTrack demonstration in vehicle tracking. Photo by Roboflow.
	This project, entitled “Ball Don’t Lie: AI Basketball Referee Using ByteTrack Algorithm and YOLOv8,” leverages the mentioned technologies to explore an innovative solution: an automated referee system designed for basketball games. By combining YOLOv8 and ByteTrack, the system aims to lessen errors induced by human referees, enhance game integrity, and foster basketball culture both in professional and amateur courts. In the long run, such a system could allow for wider adoption of AI-driven officiating tools to Taiwan’s sports leagues way beyond basketball.
## 開發目標 Objectives
	This project aims to design and implement an AI-powered basketball referee system that leverages YOLOv8 for object detection, ByteTrack with Kalman filter for multi-object tracking, and Intel OpenVINO for optimized real-time inference, in order to improve the accuracy, fairness, and accessibility of officiating in basketball games. Specifically, this project aims to:
Develop a basketball action recognition system capable of detecting player actions, namely:
Step counter: tracking instances where a player foot touches the ground for rule enforcement,
Dribble counter: tracking instances where the ball touches the ground for analysis and violation detection,
Shot counter: recognizing successful shot attempts for player scoring and performance tracking. 
Holding basketball detection: detecting player possession where both hands are holding the basketball.
Travel violation detection: detecting when a player takes more than two steps without dribbling.
Integrate the YOLOv8 model for real-time object detection of the basketball and the game players, identifying relevant body movements.
Implement the ByteTrack algorithm for multi-object tracking to ensure consistent identification of the basketball and the players across video frames.
Optimize the system with Intel OpenVINO to enable efficient, low-latency inference on CPUs and edge devices.
Design and implement a user-friendly frontend using Flet for visualization of the system in an interactive interface.
Containerize the application using Docker for portable deployment, ensuring compatibility across different environments and simplifying installation on local machines or cloud infrastructure.
## 使用技術 Technical Methods
System Architecture
	The AI basketball referee system integrates object detection, tracking, and action analysis to monitor basketball games in real time. YOLOv8 identifies the basketball and players, ByteTrack with Kalman filtering maintains consistent object identities, and the action analysis module detects rule violations and counts shots, dribbles, and steps. Results are displayed via a Flet-based frontend, and the system is containerized using Docker for portable deployment.

Figure 3.1. Overall system architecture.
Object Detection
	The YOLOv8 model, pretrained with Ultralytics and fine-tuned for basketball objects, detects the ball and players in each frame. Intel OpenVINO is used to optimize the model for low-latency inference on CPU and edge devices, enabling real-time processing without requiring a high-end GPU. Detection outputs include bounding boxes, class labels, and confidence scores, which are used by the tracking and analysis modules.
Object Tracking
To maintain consistent identities of players and the basketball across frames, the system employs the ByteTrack algorithm. Each detection is represented as d=(bbox,score), and two confidence thresholds, TH=0.6 and TL=0.1, are defined. Detections are split into high-confidence (DH​) and low-confidence (DL) sets based on these thresholds.
The tracking process is performed in two stages. In the first stage, high-confidence detections (DH) are associated with predicted tracking bounding boxes (T) using an Intersection-over-Union (IoU) calculation and the Hungarian matching algorithm. If the IoU is below 0.2, the matching is skipped. Unmatched detections are stored in Dremain​, and unmatched tracks in Tremain​ for future processing.
In the second stage, low-confidence detections (DL) are matched with remaining tracks (Tremain​) using the same matching approach. This allows the system to retain potential valid objects that may have lower detection confidence. To handle temporary object loss, unmatched tracks are placed in a cost matrix (Tcost​) calculated as 1-IoU. Tracks that remain unmatched for more than 30 frames are deleted, ensuring that only relevant objects are tracked.

Figure 3.2. ByteTrack matching algorithm code.
A Kalman filter is used to predict and smooth the trajectories of tracked objects, particularly the basketball, which can move rapidly or become temporarily occluded. The filter’s state transition model accounts for position, velocity, and acceleration:

Figure 3.3. Kalman filter’s state transition equations.
The process noise (Q) gives acceleration higher uncertainty to account for fast basketball movement, while the measurement noise (R) is adjusted dynamically based on detection confidence—high-confidence detections use smaller R values, and low-confidence detections use larger R values. The filter also integrates gravity into the vertical acceleration (ay) and applies damping to ensure convergence of the predicted trajectory.
Frame Processing
Each video frame is processed sequentially: YOLOv8 detects objects, ByteTrack and the Kalman filter track them across frames, and the action analysis module evaluates game events. The system pipeline ensures that all detections and tracking updates occur in real time, enabling accurate monitoring of basketball actions.

Figure 3.4. ByteTrack and Kalman filter frame processing diagram.
Action Analysis
The action analysis segment interprets the trajectories of the basketball and player movements to monitor game events and detect rule violations. It uses tracking data from ByteTrack and the Kalman filter, combined with pose estimation, to implement several key functions in real time.
Step Counter
Steps are counted using ankle positions from the pose tracker. A foot is considered “down” if the ankle is near the ground for several consecutive frames. Steps are incremented when alternating foot-down events are detected, i.e., when a different foot touches the ground after the previous foot.

Figure 3.5. Step counter code.
Dribble Counter
Dribbles are counted by tracking the vertical movement of the basketball. The system calculates the change in height of the basketball as y=yc-yl, the difference between the y-coordinates of the current frame (yc) and the last frame (yl). Downward motion (y>0) followed by upward motion (y<0) is considered a completed dribble, simulating the bouncing of the ball.

Figure 3.6. Dribble counter code.
Shot Counter
The shot counter evaluates whether a shot attempt is successful by analyzing the basketball’s trajectory relative to the hoop. The system identifies coordinates above and below the hoop center (xh, yh), draws a linear trajectory line for the ball, and calculates its intersection with y=yh​. If the intersection falls within the hoop’s horizontal boundaries (x1xtx2​), the system registers a goal.

Figure 3.7. Shot counter diagram.
Holding Basketball Detection
	Holding the basketball is detected by comparing the basketball’s center with the player’s hand coordinates from the pose tracker. If the ball is positioned between the hands and they are sufficiently close, the system registers that the player is holding the ball. This information is used to support travel violation detection.

Figure 3.8. Holding basketball detection Python code.
Travel Violation Detection
	Travel violations are triggered when step counts exceed two while the player is holding the basketball. The system combines holding detection and step counter outputs to determine whether a travel violation has occurred.

Figure 3.9. Travel violation detection code.

Integration and Real-Time Processing
The system integrates detection, tracking, and action analysis to operate in real time. YOLOv8 provides object detections, which are tracked across frames by ByteTrack and the Kalman filter. The tracking outputs feed directly into the action analysis modules, which count shots, dribbles, and steps, and detect violations such as traveling or holding the basketball.
Intel OpenVINO is used to optimize YOLOv8 inference for CPU-based devices, reducing latency and ensuring smooth real-time performance. The frame processing pipeline is designed for efficiency, allowing each frame to pass sequentially through detection, tracking, and analysis modules without significant delay. The system’s results are displayed through a Flet-based user interface, providing an interactive and clear view of game events. Finally, the entire application is containerized using Docker, making it portable and easy to deploy on different devices or environments, such as school gyms or community courts.

Figure 3.10. Sample snapshot of Flet-based UI of the AI basketball referee application.
## 成果展示及技術說明 Results and Implementation
	The AI basketball referee successfully performed real-time detection, tracking, and action analysis during multiple test scenarios. YOLOv8 accurately detected basketballs and players, while ByteTrack with Kalman filtering maintained stable identities even during rapid movements and brief occlusions. The system reliably counted shots, dribbles, and steps, and correctly detected travel violations. Overall, the system demonstrated high accuracy and responsiveness suitable for practical use in local basketball courts in Taiwan.
	To illustrate the system in action, demonstration videos are provided in a YouTube playlist:
https://www.youtube.com/playlist?list=PLdC83DTeBiTrQ2dVX5Xm0Vn1gJHYYx18u
	The complete project code is publicly accessible on GitHub, including detection models, tracking modules, action analysis algorithms, Flet frontend, and Docker deployment configuration:
https://github.com/CliffyangZ/AI-Baskeetball-Referee.git
The project successfully achieved its objectives, integrating YOLOv8, ByteTrack, Kalman filtering, and OpenVINO optimization to produce a functional AI basketball referee. The system is capable of real-time monitoring and analysis, and the modular design allows future enhancements, such as implementing double dribble detection or advanced player performance metrics.
