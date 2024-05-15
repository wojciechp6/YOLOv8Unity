# YOLOv8 Unity 
YOLOv8 Unity integrates cutting-edge and state-of-the-art Deep Learning models with the Unity engine using the Barracuda library. It contains examples of **Object Detection** and **Instance Segmentation**.  
   
This project is the direct continuation of my previous project [YOLO-UnityBarracuda](https://github.com/wojciechp6/YOLO-UnityBarracuda).

## YOLOv8 
[YOLOv8](https://github.com/ultralytics/ultralytics) is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of computer vision tasks.    
    
This new version of YOLO achieves better accuracy and is even faster than its predecessors. 
![image](https://github.com/wojciechp6/YOLOv8Unity/assets/29753380/7d2dd65f-1564-4be4-83c9-c2c424d31734)

## Usage
This project uses Unity 2022.3.   

### Instance Segmentation
To run segmentation you need to obtain the onnx version of the segmentation model and indicate it in the script.  
1. ~~Download the already converted model from [PINTO model zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/345_YOLOv8)~~ (not available for now) or convert it by yourself using [export command](https://docs.ultralytics.com/usage/cli/#export).
2. Copy the segmentation model to *Assets*.
3. Open *Scenes/Segmentation*.
4. Select *Main Camera*.
5. In the *Segmentator* component point your segmentation model in the *Model File* field.
6. Run the scene.

### Object Detection
To run object detection you need to obtain the onnx version of the detection model and indicate it in the script.  
1. ~~Download the already converted model from [PINTO model zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/345_YOLOv8)~~ (not available for now) or convert it by yourself using [export command](https://docs.ultralytics.com/usage/cli/#export).
2. Copy the detection model to *Assets*.
3. Open *Scenes/Detection*.
4. Select *Main Camera*.
5. In the *Detector* component point your detection model in the *Model File* field.
6. Run the scene.
