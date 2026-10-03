## Related Work

### Calibration Timing and Modality
Prior research on online automatic geometric calibration has predominantly focused on LiDAR-camera systems, leveraging the high point density of LiDAR to facilitate robust extrinsic estimation [3, 4, 6, 7, 8, 9]. A separate line of work addresses online calibration for broader multimodal sensor arrays or general multi-sensor setups, often relying on motion-based cues or moving object tracking to constrain the calibration parameters [5, 10]. In contrast, the specific challenge of online automatic geometric calibration for radar-camera pairs has been addressed by a limited number of recent studies, which aim to overcome the inherent sparsity of radar data [11, 12, 13]. While these prior radar-camera methods establish a baseline for this modality pair, our work distinguishes itself by proposing the first end-to-end online automatic geometric calibration network specifically designed to handle the unique data sparsity and height uncertainty of radar, thereby establishing a new benchmark that surpasses both previous radar-camera methods and state-of-the-art LiDAR-camera techniques.

### Feature Representation Strategy
To address the challenges of sensor sparsity and measurement noise, prior approaches have employed diverse feature representation strategies. Some methods bypass deep feature extraction entirely, relying instead on direct geometric optimization to estimate calibration parameters [12, 13]. Other deep learning-based approaches utilize single CNNs to process projected depth and RGB data [6], or employ 3D Spatial Transformer Networks to handle geometric transformations [7]. Additionally, cost volume representations have been used to encode spatial relationships for matching [9], while others utilize coarse and fine convolutional neural network features to capture multi-scale information [11]. However, no cited prior work adopts a Dual-Perspective representation that explicitly combines frontal and bird's-eye views with a Selective Fusion Mechanism. Our approach uniquely leverages the complementary strengths of these two perspectives—rich but sensitive height information in the frontal view and robust features in the bird's-eye view—to mitigate the effects of radar height uncertainty, a gap not addressed by the aforementioned single-perspective or geometric-only methods.

### Correspondence Matching Mechanism
The mechanism for establishing correspondences between modalities varies significantly across existing literature. Geometric constraint-based matching, such as RANSAC, is frequently employed in traditional calibration algorithms to enforce spatial consistency [12, 13]. In the domain of deep learning, some methods rely on implicit CNN regression to predict extrinsic parameters without explicit matching [6], while others optimize for geometric and photometric consistency [7]. Cost volume matching has also been utilized to find correspondences through iterative refinement [9]. In contrast, our paper introduces a Multi-Modal Cross-Attention Mechanism to explicitly find location correspondences through cross-modal matching. This approach allows the network to learn complex, non-linear relationships between sparse radar points and dense camera features, offering a distinct advantage over rigid geometric constraints or simple metric-based matching strategies found in prior work.

### Training Supervision Strategy
The design of the training supervision strategy is critical for ensuring robustness against noisy and sparse sensor data. Several prior methods employ supervised regression on randomly decalibrated data to train their networks [6], while others utilize self-supervised learning frameworks that do not require an explicit matcher [7]. Standard supervised losses using ground-truth correspondences are also common in calibration networks [9]. Additionally, boosting-inspired training on residual error has been used to refine calibration estimates [11]. However, none of the cited prior works implement a Noise-Resistant Matcher specifically designed to provide better supervision and enhance robustness against sparsity and height uncertainty. Our proposed Noise-Resistant Matcher addresses this gap by providing more reliable supervision signals during the training phase, ensuring that the learned features remain robust despite the significant measurement uncertainties inherent in radar data.

## References

[1] Obstacle detection using millimeter-wave radar and its visualization on image sequence
[2] Radar and vision sensor fusion for object detection in autonomous vehicle surroundings
[3] Automatic targetless LiDAR--camera calibration: a survey
[4] Automatic targetless extrinsic calibration of a 3d lidar and camera by maximizing mutual information
[5] Motion-based calibration of multimodal sensor arrays
[6] RegNet: Multimodal Sensor Registration Using Deep Neural Networks
[7] CalibNet: Geometrically Supervised Extrinsic Calibration using 3D
  Spatial Transformer Networks
[8] Calibrcnn: Calibrating camera and lidar by recurrent convolutional neural network and geometric constraints
[9] LCCNet: LiDAR and Camera Self-Calibration using Cost Volume Network
[10] Online multi-sensor calibration based on moving object tracking
[11] Targetless Rotational Auto-Calibration of Radar and Camera for
  Intelligent Transportation Systems
[12] A Continuous-Time Approach for 3D Radar-to-Camera Extrinsic Calibration
[13] Spatiotemporal Calibration of 3D Millimetre-Wavelength Radar-Camera
  Pairs