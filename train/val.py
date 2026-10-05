from ultralytics import YOLO

model = YOLO('runs/best.pt''yolov8n.pt')
# model = YOLO('yolov8n.pt')

model.val(data='a_seg_train.yaml''a_detect_train.yaml',imgsz=640,batch=4, conf=0.25,iou=0.6, workers=0,cache=True)
# model.val(data='a_detect_train.yaml',imgsz=640,batch=4, conf=0.25,iou=0.6, workers=0,cache=True)