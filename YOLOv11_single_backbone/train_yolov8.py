
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld

if __name__=="__main__":

    # 使用自己的YOLOv8.yamy文件搭建模型并加载预训练权重训练模型
    model = YOLO(r"/home/shengtuo/tangfan/YOLOv11_single_backbone/ultralytics/cfg/models/11/yolo11_MobileMamba_S6.yaml")\
        # .load(r'E:\bilibili\model\YOLOV8_new\ultralytics-main\yolo11n.pt')  # build from YAML and transfer weights

    results = model.train(data="/home/shengtuo/tangfan/YOLOv11_single_backbone/ultralytics/cfg/datasets/VOC_my.yaml",
                          epochs=300,
                          imgsz=640,
                          batch=4,
                          # cache = False,
                          # single_cls = False,  # 是否是单类别检测
                          # workers = 0,
                         # resume='',
                          amp = True
                          )

    # # 使用YOLOv8.yamy文件搭建的模型训练
    # model = YOLO(r"D:\bilibili\model\YOLOV8_new\ultralytics-main\ultralytics\cfg\models\11\yolo11_shufflenetv1.yaml")  # build a new model from YAML
    # model.train(data=r'D:\bilibili\model\ultralytics-main\ultralytics\cfg\datasets\VOC_my.yaml',
    #                       epochs=300, imgsz=640, batch=8, close_mosaic=10)
    #
    # # 加载已训练好的模型权重搭建模型训练
    # model = YOLO(r'D:\bilibili\model\ultralytics-main\tests\yolov8n.pt')  # load a pretrained model (recommended for training)
    # results = model.train(data=r'D:\bilibili\model\ultralytics-main\ultralytics\cfg\datasets\VOC_my.yaml',
    #                       epochs=100, imgsz=640, batch=4)




