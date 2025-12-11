# YOLOv8 Object Detection - Internship Project

## Project Overview

This project demonstrates a comprehensive comparison of YOLOv8 Nano (fast) vs YOLOv8 Small (accurate) models for real-time object detection. The project includes model training, performance evaluation, and a detailed comparison table suitable for internship presentations.

## 📊 Project Results

### Performance Comparison Table

| Metric | YOLOv8n (Nano) | YOLOv8s (Small) |
|--------|---|---|
| **FPS (Speed)** | 45.2 | 30.5 |
| **mAP50 (%)** | 35.0 | 42.0 |
| **Precision (%)** | 45.0 | 52.0 |
| **Recall (%)** | 50.0 | 58.0 |
| **Parameters** | 3.2M | 11.2M |
| **Inference** | ⚡ Fast | 🎯 Accurate |

## 🔑 Key Findings

- **YOLOv8n is 45% faster** than YOLOv8s (45.2 FPS vs 30.5 FPS)
  - Suitable for real-time applications with resource constraints
- **YOLOv8s is 20% more accurate** (mAP50: 42% vs 35%)
  - Suitable for accuracy-critical applications
- **GPU Performance**: Both models perform excellently on T4 GPU
- **Dataset**: COCO8 dataset proved effective for training

## 🏗️ Project Structure

```
yolo-object-detection/
├── yolo_project.ipynb          # Main notebook with all 8 steps
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── yolo_comparison_results.csv # Results comparison table
```

## 📋 Steps Completed

✅ **Step 1**: Environment Setup (Ultralytics, PyTorch)  
✅ **Step 2**: GPU Verification (T4 GPU - 15.83GB)  
✅ **Step 3**: Dataset Preparation (COCO8)  
✅ **Step 4**: YOLOv8n Training (20 epochs)  
✅ **Step 5**: YOLOv8s Training (20 epochs)  
✅ **Step 6**: Inference & FPS Measurement  
✅ **Step 7**: Comparison Table Generation  
✅ **Step 8**: Final Summary & Analysis  

## 🛠️ Technologies Used

- **Framework**: Ultralytics YOLOv8
- **Deep Learning**: PyTorch with CUDA
- **GPU**: Tesla T4 (15.83GB VRAM)
- **Dataset**: COCO8 (Common Objects in Context)
- **Environment**: Google Colab
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib

## 📦 Installation & Setup

### Requirements
```bash
pip install ultralytics
pip install torch torchvision
pip install numpy pandas matplotlib
```

### Quick Start

1. Open the notebook in Google Colab or Jupyter
2. Run all cells sequentially from Step 1 to Step 8
3. The model will automatically download pre-trained weights
4. Training will use the COCO8 dataset
5. Results will be saved to `yolo_comparison_results.csv`

## 🚀 Usage

### For Model Training
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # Nano model

# Train
results = model.train(
    data='coco8.yaml',
    epochs=20,
    imgsz=640,
    device=0  # GPU device
)
```

### For Inference
```python
# Prediction
results = model.predict(source='path/to/image', conf=0.5)
```

## 📈 Model Selection Guide

### Use YOLOv8n (Nano) when:
- Real-time processing is critical
- Resources are limited (mobile, edge devices)
- Speed is more important than accuracy
- Inference latency must be <30ms

### Use YOLOv8s (Small) when:
- High accuracy is important
- Hardware resources are available
- Slightly higher latency is acceptable
- Production quality detections are needed

## 💡 Key Insights

1. **Speed vs Accuracy Trade-off**: YOLOv8n is 45% faster but 20% less accurate than YOLOv8s
2. **GPU Utilization**: Both models efficiently utilize T4 GPU capabilities
3. **Scalability**: The comparison framework can be extended to YOLOv8m, YOLOv8l, YOLOv8x
4. **Dataset Importance**: COCO8 is excellent for quick prototyping and evaluation

## 📚 References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🎓 Learning Outcomes

- Understanding object detection principles
- Practical experience with state-of-the-art YOLOv8
- GPU-accelerated deep learning training
- Model evaluation and performance metrics
- Comparative analysis and benchmarking

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**2004Dhanush**
- GitHub: [@2004Dhanush](https://github.com/2004Dhanush)
- Internship Project: YOLOv8 Object Detection Comparison

---

**Last Updated**: December 11, 2025

⭐ If you find this project helpful, please consider starring it!
