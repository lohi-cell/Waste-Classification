# Plastic Waste Classification Using CNN

## Project Overview
This project focuses on developing a Convolutional Neural Network (CNN) to classify images of plastic waste into different categories such as Organic and Recyclable. The dataset is sourced from Kaggle, containing labeled images structured into separate folders for training and testing.

### Dataset
The dataset used is from Kaggle and can be accessed [here](https://www.kaggle.com/datasets/techsash/waste-classification-data/data). It includes:
- **TRAIN:** 22,564 images for training the model.
- **TEST:** 2,513 images for evaluating model performance.

### Tools and Libraries
Ensure the following dependencies are installed:
- **opencv-python:** For image processing and manipulation.
- **tensorflow:** For building and training the CNN model.
- **numpy:** For handling numerical operations and arrays.
- **pandas:** For data manipulation and organization.
- **matplotlib:** For visualizing data and graphs.
- **tqdm:** For displaying progress bars during data processing.

Run the following command to install all dependencies:
```bash
pip install opencv-python tensorflow numpy pandas matplotlib tqdm
```

## Modules

### Data Preprocessing
Data preprocessing involves:
1. **Loading Images:** Reading images from the dataset directory.
2. **Resizing:** Standardizing image dimensions.
3. **Color Conversion:** Converting images from BGR to RGB.
4. **Normalization:** Scaling pixel values to [0,1].
5. **Label Encoding:** Converting 'Organic' and 'Recyclable' labels into numerical values.
6. **Splitting Data:** Dividing data into training and testing sets.

### Data Visualization
To understand the dataset distribution and features, the following visualizations were used:
- **Class Distribution Pie Chart:** Displays the proportion of Organic and Recyclable waste images.
- **Image Histogram:** Analyzes pixel intensity distribution to understand color characteristics.

### Model Building
The CNN model architecture includes:
1. **Convolutional Layers (Conv2D):** Detect features like edges and textures.
2. **MaxPooling Layers:** Reduce dimensionality and prevent overfitting.
3. **Flatten Layer:** Converts 2D feature maps to 1D vectors.
4. **Dense Layers:** Fully connected layers for prediction.
5. **Dropout Layers:** Prevent overfitting by randomly disabling neurons.
6. **Activation Functions:** ReLU for hidden layers, sigmoid for output.

### Model Training
- The model was trained using **ImageDataGenerator** for rescaling and augmenting images.
- The **fit()** method was used to train the model for **7 epochs** with a **batch size of 64**.
- The model used **binary cross-entropy loss** and the **Adam optimizer**.

### Model Evaluation
- Performance was evaluated using accuracy and loss metrics.
- A **confusion matrix** confirmed high prediction accuracy with minimal misclassifications.

### Predictions
The trained model predicts whether an image belongs to the 'Organic' or 'Recyclable' category based on the probability score.

## Results
- **Accuracy:** The model achieved **95.1%** accuracy on both validation and test datasets.
- **Accuracy Graph:** Shows consistent improvement across epochs.
- **Loss Graph:** Displays steady loss reduction, indicating effective learning.
- **Confusion Matrix:** Demonstrates high precision with minimal errors.

## Performance Optimization
**Note:** It is recommended to use **Google Colab** with a **T4 GPU** for faster processing. To enable GPU in Colab:

1. Navigate to **Runtime > Change runtime type**.
2. Select **T4 GPU**.

## Usage
1. Clone the repository.
2. Install dependencies.
3. Run the preprocessing script.

## Contributing
Feel free to fork this repository and submit pull requests.

## License
This project is licensed under the **MIT License**.



