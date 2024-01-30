# YOLOv8 Unity 
YOLOv8 Unity integrates cutting-edge and state-of-the-art Deep Learning models with the Unity engine using the Barracuda library. It contains examples of **Object Detection** and **Instance Segmentation**.  
   
This project is the direct continuation of my previous project [YOLO-UnityBarracuda](https://github.com/wojciechp6/YOLO-UnityBarracuda).

## YOLOv8 
[YOLOv8](https://github.com/ultralytics/ultralytics) is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of computer vision tasks.    
    
This new version of YOLO achieves better accuracy and is even faster than its predecessors. 
![image](https://github.com/wojciechp6/YOLOv8Unity/assets/29753380/7d2dd65f-1564-4be4-83c9-c2c424d31734)

## Usage
### Instance Segmentation
To use this project you need to obtain onnx version of the segmentation model and indicate it in the script.  
1. Download the already converted version from [PINTO model zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/345_YOLOv8) or convert it by yourself using [export command](https://docs.ultralytics.com/usage/cli/).
2. Copy the segmentation model to *Assets*.
3. Open *Scenes/Sample Scene*.
4. Select *Main Camera*.
5. In the *WebCamDetector* component point your segmentation model in the *Model File* field.
6. Run the scene.

### Object Detection
1. In *WebCamDetector.cs* script code replace `YOLOv8Segmentation` class with `YOLOv8`.
2. Follow the steps from ***Instance Segmentation*** usage steps, but instead of the segmentation model use the detection model.


 
