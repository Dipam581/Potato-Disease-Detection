# Potato Disease Classification Project

## Overview
This project is a deep learning-based solution to classify potato leaves into three categories:
- **Early Blight**
- **Late Blight**
- **Healthy**

The dataset used contains labeled images of potato leaves, and a convolutional neural network (CNN) model is trained to predict the category of a given image. The model is implemented using TensorFlow and Keras.

## Features
- **Dataset Handling**: Efficient loading, preprocessing, and splitting into train, test, and validation sets.
- **CNN Architecture**: Multi-layer convolutional neural network with L1 regularization and dropout to handle overfitting.
- **Model Evaluation**: Detailed classification report and metrics like accuracy, precision, recall, and F1-score.
- **Image Prediction**: Upload and predict a new image using the trained model.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/potato-disease-classification.git
   cd potato-disease-classification
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset consists of 2,152 labeled images of potato leaves belonging to the following three classes:
- **Potato__Early_Blight**
- **Potato__Late_Blight**
- **Potato__Healthy**

### Dataset Directory Structure:
```
PotatoDataset/
  Potato__Early_Blight/
  Potato__Late_Blight/
  Potato__Healthy/
```
Ensure the dataset is placed in the `PotatoDataset` directory before running the project.

## Model Architecture
The model is built using TensorFlow's Keras API and includes the following layers:
1. **Conv2D**: Three convolutional layers with ReLU activation and filters [32, 64, 64].
2. **MaxPooling2D**: Pooling layers to reduce dimensionality.
3. **Flatten**: To convert the 2D matrix into a 1D vector.
4. **Dense**: Fully connected layers with L1 regularization and dropout.
5. **Softmax Output Layer**: For multiclass classification.

## Training the Model
Run the following command to train the model:
```bash
jupyter notebook
```

## Predicting on a New Image
To predict the class of a new image from the internet:
1. Add the URL of the image in the script.
2. Run the following script:
   ```bash
   python predict.py --url "<IMAGE_URL>"
   ```
3. The predicted class will be displayed in the terminal.

## Evaluation
The trained model is evaluated using a test dataset, and a classification report is generated:
```bash
python evaluate.py
```
Metrics include:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

## Adding Noise to Reduce Overfitting
To enhance generalization, Gaussian noise is added to the dataset during training.

## Example Usage
### Prediction Output:
```
True Label: Potato__Early_Blight
Predicted Label: Potato__Late_Blight
```

## Contributions
Contributions are welcome! Please create an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **Dataset**: [Kaggle Potato Leaf Dataset](https://www.kaggle.com/datasets)
- **Libraries Used**: TensorFlow, NumPy, Matplotlib, sklearn

---
For any questions, please feel free to reach out at [your email].

