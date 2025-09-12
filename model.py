import os
from ultralytics import YOLO

# ---- CONFIG ----
OUTPUT = "yolo_labelled"   # folder where your images, labels, and data.yaml already exist
MODEL = "yolov8s.pt"       # pretrained YOLOv8n (nano model)
EPOCHS = 5
# ----------------

# Path to your data.yaml
yaml_path = os.path.join(OUTPUT, "data.yaml")
 
# Step: Train YOLOv8
model = YOLO(MODEL)  # load pretrained YOLOv8n
results = model.train(
    data=yaml_path,
    device=0,                # GPU
    epochs=5,               # 50–100
    imgsz=1024,              # bigger for small objects
    batch=4,                 # raise if GPU allows
    optimizer="AdamW",
    lr0=0.001,               # a touch lower than default
    weight_decay=0.0005,
    mosaic=1.0,              # keep mosaic but...
    close_mosaic=10,         # last 10 epochs without mosaic for precise boxes
    degrees=0.0,             # keep geometry mild
    shear=0.0,
    perspective=0.0,
    scale=0.3,               # light scale
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    mixup=0.1,
    patience=20,             # early stop if plateaus
    cache=True,
    workers=0                # set >0 if dataloader issues are solved on your OS
)

# Step: Evaluate on validation set
metrics = model.val()
print("📊 Validation metrics:", metrics)





