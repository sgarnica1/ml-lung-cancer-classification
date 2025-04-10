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

After that, all images are procces and loaded into a `pandas` `DataFrame` for better data manipulation.

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
