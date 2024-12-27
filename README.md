# Image Classification Project

## Overview

This project implements an image classification system using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. The system classifies images into 10 categories, including airplanes, cars, birds, and more. This project demonstrates the application of deep learning for visual data classification tasks.

**Note**: This project is for educational purposes only.

## Features

- **CIFAR-10 Dataset**: Leverages a well-known dataset for image classification tasks.
- **Convolutional Neural Network**: Implements a deep learning model using TensorFlow/Keras for high accuracy.
- **Modular Codebase**: Divides training and testing functionality into separate scripts.
- **Model Persistence**: Saves and loads trained models for future use.

## File Structure

1. **`main.py`**:
   - Contains the training pipeline, including dataset preprocessing, model training, and saving the trained model.

2. **`test.py`**:
   - Loads the trained model and evaluates its performance on test data.

3. **`cifar10_cnn_model.h5`**:
   - Pre-trained CNN model saved in HDF5 format for reuse.

4. **`requirements.txt`**:
   - Lists all dependencies required to run the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RaviKunapareddy/Image-Classifier.git
   cd Image-Classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train the Model
1. Ensure the CIFAR-10 dataset is available (it will be downloaded automatically by TensorFlow if not already present).
2. Run the training script:
   ```bash
   python main.py
   ```

### Test the Model
1. Load the pre-trained model and evaluate it on test data:
   ```bash
   python test.py
   ```

### Example Command
```bash
python main.py --epochs 50 --batch_size 32
python test.py --model cifar10_cnn_model.h5
```

## Example Outputs

- **Input**: An image of an Ship.  
  **Output**: Predicted label: `Ship`  
  ![Ship](https://github.com/RaviKunapareddy/Image-Classifier/blob/main/output1.png)

- **Input**: An image of a Truck.  
  **Output**: Predicted label: `Truck`  
  ![Truck](https://github.com/RaviKunapareddy/Image-Classifier/blob/main/output2.png)

## Technologies Used

- **Languages**: Python
- **Libraries**: TensorFlow, Keras, NumPy, Matplotlib

## Future Enhancements

- Extend the model to classify images from additional datasets (e.g., ImageNet).
- Improve model accuracy using advanced architectures like ResNet or EfficientNet.
- Integrate a GUI or web interface for real-time image classification.

## License

This project does not currently have a license. It is for educational purposes only, and its use or distribution should be done with the creator's consent.

## Contact

Created by **[Raviteja Kunapareddy](https://www.linkedin.com/in/ravitejak99/)**. Connect for collaboration or questions!

