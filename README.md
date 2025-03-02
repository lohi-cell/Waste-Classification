

# 🌿 **Plastic Waste Classification Using CNN**  

## 🌟 **Project Overview**  
This project focuses on developing a **Convolutional Neural Network (CNN)** to classify images of plastic waste into different categories such as ♻️ **Recyclable** and 🌱 **Organic**. The dataset is sourced from **Kaggle**, containing labeled images structured into separate folders for training and testing.  

🔗 **Live Demo:** [Try it here 🚀](https://waste-classification-kk3cuwisywcyphylpgh9o2.streamlit.app/#77af9153)  
---

### 📂 **Dataset**  
The dataset used is from Kaggle and can be accessed [here](https://www.kaggle.com/datasets/techsash/waste-classification-data/data). It includes:  
- 🏋️‍♂️ **TRAIN:** 22,564 images for training the model.  
- 🧪 **TEST:** 2,513 images for evaluating model performance.  

---

### 🛠️ **Tools and Libraries**  
Ensure the following dependencies are installed:  
- 🖼️ **opencv-python:** For image processing and manipulation.  
- 🤖 **tensorflow:** For building and training the CNN model.  
- 🔢 **numpy:** For handling numerical operations and arrays.  
- 📊 **pandas:** For data manipulation and organization.  
- 📈 **matplotlib:** For visualizing data and graphs.  
- ⏳ **tqdm:** For displaying progress bars during data processing.  

💡 **Install all dependencies:**  
```bash
pip install opencv-python tensorflow numpy pandas matplotlib tqdm
```

---

## 🧩 **Modules**  

### 🏗️ **Data Preprocessing**  
Data preprocessing involves:  
1. 🖼️ **Loading Images:** Reading images from the dataset directory.  
2. 🔄 **Resizing:** Standardizing image dimensions.  
3. 🎨 **Color Conversion:** Converting images from BGR to RGB.  
4. 🌈 **Normalization:** Scaling pixel values to [0,1].  
5. 🔢 **Label Encoding:** Converting 'Organic' and 'Recyclable' labels into numerical values.  
6. ✂️ **Splitting Data:** Dividing data into training and testing sets.  

---

### 📊 **Data Visualization**  
Visualizations to understand the dataset distribution and features:  
- 🥧 **Class Distribution Pie Chart:** Displays the proportion of Organic and Recyclable waste images.  
- 🌄 **Image Histogram:** Analyzes pixel intensity distribution to understand color characteristics.  

---

### 🏗️ **Model Building**  
The **CNN model architecture** includes:  
- 🧱 **Convolutional Layers (Conv2D):** Detect features like edges and textures.  
- 🏞️ **MaxPooling Layers:** Reduce dimensionality and prevent overfitting.  
- 🏭 **Flatten Layer:** Converts 2D feature maps to 1D vectors.  
- 🧮 **Dense Layers:** Fully connected layers for prediction.  
- 💧 **Dropout Layers:** Prevent overfitting by randomly disabling neurons.  
- ⚡ **Activation Functions:** ReLU for hidden layers, sigmoid for output.  

---

### 🚀 **Model Training**  
- 🧪 Trained using **ImageDataGenerator** for rescaling and augmenting images.  
- 🏃 **fit()** method used for training the model for **7 epochs** with a **batch size of 64**.  
- 🧮 **Loss Function:** Binary cross-entropy loss.  
- ⚡ **Optimizer:** Adam optimizer.  

---

### 🏆 **Model Evaluation**  
- 📈 **Accuracy & Loss Metrics:** Evaluated using graphs for accuracy and loss.  
- ✅ **Confusion Matrix:** Confirmed high prediction accuracy with minimal misclassifications.  

---

### 🔍 **Predictions**  
The trained model predicts whether an image belongs to:  
- 🌱 **Organic**  
- ♻️ **Recyclable**  

---

## 🎯 **Results**  
- 📊 **Accuracy:** Achieved **95.1%** accuracy on both validation and test datasets.  
- 📈 **Accuracy Graph:** Shows consistent improvement across epochs.  
- 📉 **Loss Graph:** Displays steady loss reduction, indicating effective learning.  
- 🟩 **Confusion Matrix:** Demonstrates high precision with minimal errors.  

---

## ⚡ **Performance Optimization**  
💡 **Recommended:** Use **Google Colab** with a **T4 GPU** for faster processing.  

🔧 **To enable GPU in Colab:**  
1. Navigate to **Runtime > Change runtime type**.  
2. Select **T4 GPU**.  

---

## 🚀 **Usage**  
1. 🔗 Clone the repository.  
2. 📦 Install dependencies.  
3. 🏃 Run the preprocessing script.  

---

