from ultralytics import YOLO

def main():
    # Load the lightweight YOLOv8 nano model (optimized for edge devices)
    model = YOLO('yolov8n.pt') 

    # Train the model on the Roboflow dataset
    results = model.train(
        data='data.yaml', 
        epochs=50,       
        imgsz=640,       
        batch=4,         
        device=0,        
        workers=0,
        plots=False,
        name='pest_detection_model'
    )
    
    print("Training complete!")

if __name__ == '__main__':
    main()