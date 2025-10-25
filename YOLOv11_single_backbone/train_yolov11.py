
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld

if __name__=="__main__":

    # 使用自己的yamy文件搭建模型并加载预训练权重训练模型
    model = YOLO(r"/content/drive/MyDrive/YOLOv11_single_backbone/ultralytics/cfg/models/11/yolo11_FRFN.yaml")\
        .load(r"/content/drive/MyDrive/YOLOv11_single_backbone/YOLOv11_single_backbone/yolo11n.pt")  # build from YAML and transfer weights

    results = model.train(data=r'/content/drive/MyDrive/VisDrone_Dataset/visdrone.yaml',
                          epochs=100,
                          imgsz=640,
                          batch=8,
                          )





