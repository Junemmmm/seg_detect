'''
实例分割训练
'''
from ultralytics import YOLO

#train （网络模型）
model = YOLO('yolov8s-seg.pt')  # build from YAML and transfer weights

# Train the model （数据加载的模型）
model.train(data='./a_seg_train.yaml', epochs=100, imgsz=640,batch=4, workers=0,cache=True)

