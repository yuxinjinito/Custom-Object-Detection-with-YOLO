from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Load a pretrained YOLO11n model 'best.pt'
    model = YOLO("E:/dts101 pj/resources/cfg/models/yolo11_model_10class.yaml").load("best.pt")

    # Train the model for 50 epochs with batch size 16, image size 640, device 0, optimizer Adam, learning rate 0.001, patience 10
    results = model.train(data="E:/dts101 pj/resources/cfg/datasets/yolo11_10class.yaml", epochs=50, batch=16, imgsz=640, device=0, optimizer="Adam", lr0=0.001, patience=10)