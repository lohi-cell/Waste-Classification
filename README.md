# Plastic Waste Classification Using CNN
To develop a CNN model to classify images of plastic waste into different categories.

## Week 1
## Overview
This week focuses on setting up the dataset and understanding the basic preprocessing steps required for training the model. The dataset is sourced from Kaggle and is utilized to classify waste images into categories such as organic and recyclable waste.

## Dataset
The dataset used is from Kaggle.The dataset contains labeled images of various plastic waste categories. You can use public datasets such as:
https://www.kaggle.com/datasets/techsash/waste-classification-data/data


## About the dataset
This dataset consists of labeled images of plastic waste categorized as **Organic** and **Recyclable**. It is structured into separate folders for training and testing.

- **TRAIN**: Contains images for training the model.(22,564 images )
- **TEST**: Contains images for evaluating model performance.(2,513 images)

## System Architecture
![CNN System Architecture](https://github.com/lohi-cell/Waste-Classification/blob/90e2ada64de540e77d107ab7c62e48919e1e2254/images/SystemDesign.jpg)


## Installations
Ensure the necessary dependencies are installed before proceeding. These include:
- **opencv-python**: For image processing and manipulation.
- **tensorflow**: For building and training the CNN model.
- **numpy**: For handling numerical operations and arrays.
- **pandas**: For data manipulation and organization.
- **matplotlib**: For visualizing data and graphs.
- **tqdm**: For displaying progress bars during data processing.

Run the following command to install all dependencies:
```sh
pip install opencv-python tensorflow numpy pandas matplotlib tqdm
```

## Modules
- **Data Preprocessing**: Preparing images for input into the CNN model.
- **Data Visualization**: Exploring the dataset through visual representations.
- **Model Building**: Setting up the CNN architecture for classification.

## Data Preprocessing
Data preprocessing involves the following steps:
1. **Loading Images**: Read images from the dataset directory.
2. **Resizing**: Resize images to a uniform shape suitable for the model.
3. **Color Conversion**: Convert images from BGR to RGB format.
4. **Normalization**: Scale pixel values to the range [0,1] to improve model performance.
5. **Label Encoding**: Convert categorical labels (e.g., 'Organic', 'Recyclable') into numerical values.
6. **Splitting Data**: Divide the dataset into training and testing sets.

## Data Visualization
To understand the dataset distribution and features, we use the following visualizations:
1. **Class Distribution Pie Chart**
   - A pie chart is plotted to display the proportion of **Organic** and **Recyclable** waste images in the dataset.

2. **Image Histogram**
   - A histogram is used to analyze the pixel intensity distribution across an image, helping to understand the color characteristics of the dataset.

## Conclusion
This week covers the preprocessing of the dataset to ensure it is in the correct format for training a CNN model. The preprocessing steps include resizing images, normalizing pixel values, and encoding labels to facilitate effective model training.

## Week 2
## Overview
In Week 2, we focused on building and training a Convolutional Neural Network (CNN) to classify plastic waste images into two categories: **Organic** and **Recyclable**. The model was built using TensorFlow and Keras.

## Convolutional Neural Network (CNN)
A **Convolutional Neural Network (CNN)** is a type of deep learning algorithm used mainly for image analysis. It consists of multiple layers:  
1. **Convolutional layers** to detect features (edges, shapes, etc.) from images.
2. **Pooling layers** to downsample the feature maps and reduce dimensionality.
3. **Fully connected layers** to classify or predict based on the extracted features.  
CNNs are highly efficient in tasks like image classification, object detection, and facial recognition due to their ability to automatically extract and learn relevant features.

![Image Alt Text](https://github.com/lohi-cell/Waste-Classification/blob/90e2ada64de540e77d107ab7c62e48919e1e2254/images/CNN-based-waste-management-Model.jpg)

## Key Modules

## 1. **Sequential Model**
The CNN model is built using the **Sequential** API. It consists of:
- **Conv2D Layers**: These layers detect features like edges and textures in images.
- **MaxPooling Layers**: Reduce the size of the feature maps to speed up training and prevent overfitting.
- **Flatten Layer**: Converts the 2D feature maps to a 1D vector for the fully connected layers.
- **Dense Layers**: Fully connected layers make predictions based on the learned features.
- **Dropout Layers**: Helps prevent overfitting by randomly disabling neurons during training.
- **Activation Functions**: ReLU for hidden layers and sigmoid for the output layer to predict binary categories.

## 2. **ImageDataGenerator**
Used for **data augmentation** and **rescaling** images:
- **Training**: Rescale pixel values to [0,1] for the model.
- **Testing**: Same rescaling applied to the test data.

## 3. **Model Training**
- **fit()** method is used to train the model for 10 epochs.
- The model uses **binary cross-entropy** loss and the **Adam optimizer** for training.
- **Batch size** is set to 64, and **accuracy** is tracked during training.

The model was trained using an `ImageDataGenerator` to efficiently load and preprocess images. Images were rescaled to the range [0, 1] to improve model performance. The model architecture consists of convolutional layers for feature extraction, max pooling for dimensionality reduction, and dense layers for classification.  

The model was trained for 10 epochs using the Adam optimizer and binary cross-entropy loss function.

## 4. **Model Evaluation**
After training, the model's performance is evaluated on test data, measuring **accuracy** and **loss**.Performance was evaluated on a validation set using the accuracy metric.

## 5. **Prediction**
The trained model predicts the probability of an input image belonging to either the 'Organic' or 'Recyclable' class. The class with the higher probability is the predicted outcome.

## Conclusion
This week, we built a CNN model to classify plastic waste images. The model was trained, evaluated, and is now capable of predicting new images into the appropriate categories.

## Performance Optimization
**Note:** It is recommended to use Google Colab with a **T4 GPU** for faster processing. To enable GPU in Colab:
1. Navigate to **Runtime > Change runtime type**.
2. Select **T4 GPU**.

## Usage
1. Clone the repository.
2. Install dependencies.
3. Run the preprocessing script.

## Contributing
Feel free to fork this repository and submit pull requests.

## License
This project is licensed under the MIT License.

