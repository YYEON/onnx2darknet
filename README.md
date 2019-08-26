# onnx2darknet
> Convert yolo2 model of ONNX format to darknet format

## Get ONNX format
* 

## Convert ONNX format -> Darknet format
* __python onnx2darknet.py 'yolov2.onnx' 'yolov2.cfg' 'yolov2.weights'__

## Inference Test(Darknet framework)
### try predict
* Darknet Framework installation
```python
  git clone https://github.com/pjreddie/darknet
  cd darknet
  make 
```
* Run detector
  * ./darknet detect cfg/yolov2.cfg yolov2.weights data/dog.jpg

## Currently supported
### Models
* Yolov2-tiny

### Operator
* BatchNormalization
* Conv
* LeakyRelu
* MaxPool

## Requirements and Develop Environment
* python >= 2.7
* onnx >= 1.2.1
* darknet 

## Reference
* Using ONNX for Inference (YOLO2): [https://github.com/purelyvivid/yolo2_onnx]
* Detection Using A Pre-Trained Model: [https://pjreddie.com/darknet/yolo/]
