# Plastic Waste Classification Using CNN
To develop a CNN model to classify images of plastic waste into different categories.

## Week 1
### Overview
This week focuses on setting up the dataset and understanding the basic preprocessing steps required for training the model. The dataset is sourced from Kaggle and is utilized to classify waste images into categories such as organic and recyclable waste.

# Dataset
The dataset used is from Kaggle.The dataset contains labeled images of various plastic waste categories. You can use public datasets such as:
https://www.kaggle.com/datasets/techsash/waste-classification-data/data


# About the dataset
This dataset consists of labeled images of plastic waste categorized as **Organic** and **Recyclable**. It is structured into separate folders for training and testing.

- **TRAIN**: Contains images for training the model.(22564 images )
- **TEST**: Contains images for evaluating model performance.(2513 images)

### Installations
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

### Modules
- **Data Preprocessing**: Preparing images for input into the CNN model.
- **Data Visualization**: Exploring the dataset through visual representations.
- **Model Building**: Setting up the CNN architecture for classification.

### Data Preprocessing
Data preprocessing involves the following steps:
1. **Loading Images**: Read images from the dataset directory.
2. **Resizing**: Resize images to a uniform shape suitable for the model.
3. **Color Conversion**: Convert images from BGR to RGB format.
4. **Normalization**: Scale pixel values to the range [0,1] to improve model performance.
5. **Label Encoding**: Convert categorical labels (e.g., 'Organic', 'Recyclable') into numerical values.
6. **Splitting Data**: Divide the dataset into training and testing sets.

### Data Visualization
To understand the dataset distribution and features, we use the following visualizations:
1. **Class Distribution Pie Chart**
   - A pie chart is plotted to display the proportion of **Organic** and **Recyclable** waste images in the dataset.

2. **Image Histogram**
   - A histogram is used to analyze the pixel intensity distribution across an image, helping to understand the color characteristics of the dataset.

### Explanation
This week covers the preprocessing of the dataset to ensure it is in the correct format for training a CNN model. The preprocessing steps include resizing images, normalizing pixel values, and encoding labels to facilitate effective model training.

## Usage
1. Clone the repository.
2. Install dependencies.
3. Run the preprocessing script.

## Contributing
Feel free to fork this repository and submit pull requests.

## License
This project is licensed under the MIT License.

