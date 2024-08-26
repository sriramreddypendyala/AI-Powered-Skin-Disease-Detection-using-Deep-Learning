# AI-Powered Skin Disease Detection Using Deep Learning

## Overview

This project aims to develop a deep learning model for the detection of skin diseases using images. The model leverages Convolutional Neural Networks (CNNs) to classify various skin conditions from images, providing a tool for early diagnosis and support for dermatological assessments.

## Features

- **Image Classification:** Classifies images into one of seven skin disease categories.
- **Data Augmentation:** Enhances model performance by artificially enlarging the training dataset.
- **Web Interface:** Interactive web application built with Streamlit for real-time predictions.
- **High Accuracy:** Utilizes advanced deep learning techniques for improved classification accuracy.

## Dataset

The project uses the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) which includes images of skin lesions with various labels. The dataset is used to train and test the model for accurate skin disease prediction.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/sriramreddypendyala/skin-disease-detection.git
   cd skin-disease-detection
   ```

2. **Install Dependencies:**

   Install the required libraries using:

   ```bash
   pip install tensorflow keras opencv-python-headless streamlit numpy pandas matplotlib seaborn pillow scikit-learn
   ```

## Usage

### Training the Model

1. **Prepare the Data:**

   Ensure the dataset is in the correct path as specified in the code.

2. **Run Training Script:**

   Execute the following command to train the model:

   ```bash
   python train_model.py
   ```

   This will save the trained model as `Model.h5`.

### Running the Web Application

1. **Start the Streamlit App:**

   Run the following command to start the web application:

   ```bash
   streamlit run app.py
   ```

2. **Upload an Image:**

   Navigate to the Streamlit app in your browser and upload an image to get a skin disease prediction.

## Code Explanation

- **Data Visualization:**
  - Utilizes `seaborn` and `matplotlib` to visualize dataset distributions.

- **Data Preprocessing:**
  - Reads and preprocesses the image data using `PIL` and `cv2`.

- **Model Architecture:**
  - Defines a Convolutional Neural Network (CNN) using `tensorflow.keras` with layers like `Conv2D`, `MaxPooling2D`, `Dropout`, and `Dense`.

- **Training and Evaluation:**
  - Trains the model using `model.fit()` and evaluates its performance.

- **Web Application:**
  - Implements a web interface using `Streamlit` to allow users to upload images and receive predictions.

## Files

- **`train_model.py`:** Script for training the deep learning model.
- **`app.py`:** Streamlit application for real-time predictions.
- **`Model.h5`:** Trained model file.

## Contributing

Feel free to fork the repository, create a pull request, or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
