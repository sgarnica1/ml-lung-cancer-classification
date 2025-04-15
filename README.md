# Deep Learning Model for Lung Cancer Classification
This is a depp learning model for lung cancer images classification trained with a dataset retrieved from Kaggle and originally published in arXiv, a free distribution service and open-source achive. 

## Dataset
The original dataset was published by arXiv.org and consists of 25,000 images with 5 different classes of 5,000 images each and two main subclasses: colon and lung images. 

For this project, only the **lung images** where used. The dataset for lung images contains the following classes with 5,000 images each:
| Class                   | Type            | Number of Images |
| ----------------------- | --------------- | ---------------- |
| Squamous Cell Carcinoma | Malignant Tumor | 5,000            |
| Adenocarcinomas         | Malignant Tumor | 5,000            |
| Bening lung tissues     | Not Cancer      | 5,000            |


<br>

All images are 768 x 769 pixels in size and in jpeg format.

<br>

You can download the dataset from:
- [Kaggle dataset](https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images/data) (used in this project).
- [arXiv.org publication](https://arxiv.org/abs/1912.12142v1)
- [GitHub repository](https://github.com/tampapath/lung_colon_image_set)

## Project File Structure
<pre><code>
ml-lung-cancer-classification/
├── README.md
├── requirements.txt
├── lung_cancer.ipynb
├── dataset/
│   └── input/
│       └── adenonacarcinoma/
│       └── benign/
│       └── squamous_cell_carcinoma/
└── .gitignore
</code></pre>

## Data preprocessing and data splitting

### Data loading
The data is loaded from the `dataset_path` variable, which is set to the `/dataset/input` directory.

After that, all images are proccesed and loaded into a `pandas` `DataFrame` for better data manipulation.

### Data splitting
The data is then splitted into three groups as follows:
| Dataset Split | Percentage | Number of Images |
| ------------- | ---------- | ---------------- |
| Training      | 60%        | 9,000            |
| Validation    | 20%        | 3,000            |
| Testing       | 20%        | 3,000            |


### Image Data Generator

For smoother training and improved model performance, the `ImageDataGenerator` class from the `tensorflow.keras.preprocessing.image` module was utilized. This class enables efficient preprocessing and real-time data augmentation, which helps the model generalize better to unseen data.

The following data generators were created:

- **Train Data Generator**: Used to feed augmented images to the model during training, helping reduce overfitting.
- **Validation Data Generator**: Used to evaluate the model's performance on unseen (non-augmented) data during training.
- **Test Data Generator**: Used to assess the final performance of the trained model on completely unseen data after training is complete.

#### Benefits of Using an Image Data Generator

Using `ImageDataGenerator` offers several advantages:

- **Real-time Data Augmentation**: Applies random transformations (e.g., rotation, zoom, flipping) on the fly, increasing dataset diversity without increasing memory usage.
- **Memory Efficiency**: Loads and processes data in batches, which is essential when working with large datasets.
- **Improved Generalization**: By exposing the model to varied data through augmentation, it becomes more robust and less likely to overfit.
- **Built-in Preprocessing**: Easily apply rescaling, normalization, and other preprocessing steps directly within the generator.
- **Seamless Integration with Keras**: Easily plugs into the `.fit()` or `.fit_generator()` methods for model training.

#### Applied Data Augmentation Techniques
To enhance the training data and improve the model's ability to generalize, the following augmentation techniques were applied:

- `rescale=1.0 / 255.0`: Normalizes pixel values to the range `[0, 1]`, which helps in faster and more stable training.
- `rotation_range=20`: Randomly rotates images up to `20 degrees`, helping the model become invariant to orientation.
- `width_shift_range=0.2`: Randomly shifts images horizontally by up to `20%` of the width.
- `height_shift_range=0.2`: Randomly shifts images vertically by up to `20%` of the height.
- `zoom_range=0.2`: Randomly zooms in on images by up to `20%`, allowing the model to learn from different scales.
- `horizontal_flip=True`: Randomly flips images horizontally, useful for symmetry-invariant features.

These transformations are applied only during training to simulate real-world variability. The validation and test sets are rescaled but left unaltered to evaluate model performance on consistent data.

## Model Architecture

### First Model

The initial model was a straightforward **Convolutional Neural Network (CNN)** implemented using Keras' `Sequential` API. This architecture was designed to serve as a baseline for image classification of lung CT scans into benign and malignant categories.

The model architecture consists of the following layers:

- `Conv2D`: Applies 32 filters of size 3x3 with ReLU activation to detect low-level features such as edges and textures.
- `MaxPooling2D`: Downsamples feature maps to reduce spatial dimensions and computation, while retaining important information.
- `Conv2D`: A second convolutional layer with 64 filters to capture more complex patterns and structures.
- `MaxPooling2D`: Further reduces the spatial size and helps in generalization.
- `Flatten`: Converts the 2D feature maps into a 1D feature vector for the dense layers.
- `Dense`: A fully connected layer with 128 neurons and ReLU activation, serving as a high-level feature aggregator.
- `Dense`: Output layer with softmax activation to generate probability scores for each class (benign or malignant).

This model was compiled using the **categorical cross-entropy loss** function and optimized with **RMSprop** (learning rate = `2e-5`). The evaluation metrics included accuracy, precision, and recall.

### Upgraded Model
### Final Model

## Results

### First Model Performance

After training for **10 epochs**, the first model demonstrated solid performance on both the validation and test datasets. Below are the key metrics:

- **Training Accuracy**: 89.83%
- **Validation Accuracy**: 89.03%
- **Test Accuracy**: 89.50%
- **Training Loss**: 0.2424
- **Validation Loss**: 0.2406

These results indicate that the model was able to generalize well without significant overfitting. The training and validation losses are closely aligned, and the test accuracy confirms the model's ability to correctly classify unseen CT scan images with high reliability.

### Evaluation Metrics

To further assess the performance of the model, we computed detailed classification metrics using the test dataset. The following confusion matrix and classification report provide insights into how well the model performed for each class:

#### Confusion Matrix

The confusion matrix below shows the number of correct and incorrect predictions made by the model for each class:

![Confusion Matrix First Model](/assets/model_1.png)


#### Classification Report

| Class                  | Precision | Recall | F1-Score | Support |
| ---------------------- | --------- | ------ | -------- | ------- |
| **Adenocarcinoma (0)** | 0.91      | 0.78   | 0.84     | 1000    |
| **Benign (1)**         | 0.98      | 0.96   | 0.97     | 1000    |
| **Squamous Cell (2)**  | 0.82      | 0.96   | 0.89     | 1000    |
| **Accuracy**           |           |        | **0.90** | 3000    |
| **Macro Avg**          | 0.91      | 0.90   | 0.90     | 3000    |
| **Weighted Avg**       | 0.91      | 0.90   | 0.90     | 3000    |

These metrics indicate that:
- The model performs **exceptionally well** in identifying benign samples (precision = 0.98, recall = 0.96).
- It struggles slightly with differentiating between adenocarcinoma and squamous cell carcinoma, which may be due to visual similarity in some CT scan features.
- The overall **accuracy** is **90%**, with a balanced **macro average F1-score of 0.90**, confirming the model’s robustness across classes.

This detailed evaluation supports the conclusion that the model has strong generalization capabilities and can serve as a solid foundation for further improvement.

> The support column in the classification report represents the number of true instances (or samples) for each class in the test dataset. It's essentially the size of the test set for each class, which helps to give context to the precision, recall, and F1-score.



### Upgraded Model Performance
### Final Model Performance

## Bibliography
For the development of this project, the following article served as a key reference and foundation:

- Deep learning-based approach to diagnose lung cancer using CT-scan image
Yasser Khan, M.M. Kamruzzaman, et al.
https://doi.org/10.1016/j.ibmed.2024.100188

This state-of-the-art article presents a cutting-edge deep learning approach for diagnosing lung cancer using CT scan images. It provided critical insights into model architecture, image preprocessing, and evaluation strategies. The methodology outlined in this research significantly influenced the dataset preparation, data augmentation techniques, and model design choices in this project.