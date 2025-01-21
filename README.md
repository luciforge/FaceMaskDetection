
# Face Mask Detection with MobileNetV2

This project implements a face mask detection system using MobileNetV2, TensorFlow, and OpenCV. The system detects faces in real-time and classifies them into three categories: 
- `with_mask`
- `without_mask`
- `mask_weared_incorrect`

The project includes training, evaluation, and real-time detection functionalities and is designed to be portable and easy to deploy.

---

## Features
- **Model Training**: A MobileNetV2-based model is trained on a labeled dataset for multi-class classification.
- **Evaluation Metrics**: Precision, Recall, F1-Score, and Accuracy are calculated and logged.
- **Real-Time Detection**: Detects faces using OpenCV and classifies mask usage.
- **Configurable**: Paths and settings are managed via a `config.json` file for easy customization.

---

## Prerequisites
### **1. System Requirements**
- Python 3.9+
- Operating System: Windows, macOS, or Linux

### **2. Libraries**
Install the required Python libraries using:
```bash
pip install -r requirements.txt
```

Required libraries:
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

---

## Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
```

### **2. Prepare the Dataset**
Place your dataset in the following directory structure:
```
data/
├── annotations/    # XML annotation files
├── images/         # Corresponding images
```

### **3. Configure Paths**
Update the `config.json` file with your dataset and model paths:
```json
{
    "annotations_dir": "./data/annotations",
    "images_dir": "./data/images",
    "model_file": "face_mask_model.keras",
    "log_file": "evaluation_metrics.txt"
}
```

### **4. Train the Model**
Run the script to train the model:
```bash
python realtime.py
```

---

## Real-Time Detection
Run the real-time detection system using a webcam:
```bash
python realtime.py
```
Press Ctrl+C in terminal to quit.

---

## File Structure
```
.
├── config.json             # Configuration file for paths and settings
├── realtime.py             # Main script for training and detection
├── requirements.txt        # Required Python libraries
├── data/                   # Dataset folder
│   ├── annotations/        # XML annotation files
│   ├── images/             # Corresponding images
├── face_mask_model.keras   # Saved model file (after training)
├── evaluation_metrics.txt  # Metrics log file
```

---

## Key Functions
### **Training**
The model is trained using MobileNetV2 with pre-trained weights and fine-tuned on the dataset.

### **Evaluation Metrics**
The following metrics are calculated:
- Precision
- Recall
- F1-Score
- Accuracy

Metrics are saved to `evaluation_metrics.txt` for offline analysis.

### **Real-Time Detection**
The real-time detection system uses:
- OpenCV's Haar Cascade for face detection.
- The trained model for mask classification.

---

## Customization
1. **Dataset**:
   - Replace the `data/annotations` and `data/images` directories with your own dataset.
2. **Configuration**:
   - Update paths in `config.json` to match your dataset and desired file locations.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- **Dataset**: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) by Andrew Mvd on Kaggle.
- **MobileNetV2**: TensorFlow's pre-trained MobileNetV2 model.

---

## Contact
For questions or support, please reach out to [your-email@example.com].
