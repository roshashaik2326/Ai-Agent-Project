# Dog and Cat Classifier: Extended Documentation

---

## 1. Project Content

The Dog and Cat Classifier project demonstrates the use of deep learning for image classification. It focuses on distinguishing between images of dogs and cats using convolutional neural networks (CNNs), with an emphasis on practical deployment and interpretability.

**Included Components:**
- Image data preprocessing and augmentation scripts
- CNN model definition and training scripts
- Prediction and evaluation modules
- Streamlit web app for interactive classification
- Experiment and result notebooks

---

## 2. Project Code

The codebase is structured to ensure clarity, modularity, and ease of experimentation:

- **data_loader.py**  
  Loads, preprocesses, and augments dog and cat images. Handles resizing, normalization, and optionally applies data augmentation like flips or rotations.

  _Example:_  
  ```python
  from tensorflow.keras.preprocessing.image import ImageDataGenerator

  datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
  train_gen = datagen.flow_from_directory(
      'dataset/',
      target_size=(128, 128),
      batch_size=32,
      class_mode='binary',
      subset='training'
  )
  val_gen = datagen.flow_from_directory(
      'dataset/',
      target_size=(128, 128),
      batch_size=32,
      class_mode='binary',
      subset='validation'
  )
  ```
  > This snippet prepares image batches for training and validation.

- **cnn_model.py**  
  Defines the CNN architecture for image classification.

  _Example:_  
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

  model = Sequential([
      Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
      MaxPooling2D(2,2),
      Flatten(),
      Dense(64, activation='relu'),
      Dense(1, activation='sigmoid')
  ])
  ```
  > A basic CNN structure for binary classification (dog vs. cat).

- **train.py**  
  Trains the CNN model on the dataset, monitors accuracy, and saves the best model.

  _Example:_  
  ```python
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(train_gen, validation_data=val_gen, epochs=10)
  ```
  > This code compiles and trains the model, tracking performance on validation data.

- **predict.py**  
  Loads saved models and predicts new images as dog or cat.

  _Example:_  
  ```python
  from tensorflow.keras.preprocessing import image
  import numpy as np

  img = image.load_img('test.jpg', target_size=(128,128))
  img_array = image.img_to_array(img) / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  prediction = model.predict(img_array)
  print('Dog' if prediction[0][0] > 0.5 else 'Cat')
  ```
  > This predicts whether an input image is a dog or a cat.

- **app.py**  
  Streamlit web app for uploading images and viewing predictions in the browser.

---

## 3. Key Technologies

- **Python 3.x:** Core scripting language.
- **TensorFlow/Keras:** For model building and training.
- **ImageDataGenerator:** For preprocessing and augmentation.
- **NumPy:** For efficient numerical operations.
- **Streamlit:** For interactive web app deployment.
- **Matplotlib/Seaborn:** For training metric visualization.

---

## 4. Description

This project tackles the classic problem of classifying images of dogs and cats using deep learning. It leverages a convolutional neural network (CNN) for learning visual features. The workflow includes:

1. **Data Preparation:** Images are loaded, resized, normalized, and augmented to improve generalization.
2. **Model Training:** A CNN is trained to minimize binary cross-entropy loss, optimizing for high accuracy.
3. **Prediction:** New, unseen images can be classified as a dog or a cat.
4. **Deployment:** A Streamlit app lets users interactively upload and classify images.

Practical use cases include learning about basic deep learning, experimenting with image data, and demonstrating model deployment.

---

## 5. Output

### Model Performance

- **Accuracy:** Achieves ~98% on validation data with well-augmented datasets.
- **Loss Curves:** Training and validation accuracy/loss are plotted to assess overfitting or underfitting.

### Example Predictions

- **Dog Image:**  
  ![Dog Sample](images/sample_dog.png)  
  Predicted: Dog | Actual: Dog

- **Cat Image:**  
  ![Cat Sample](images/sample_cat.png)  
  Predicted: Cat | Actual: Cat

### Streamlit App

- Users can upload images and instantly receive model predictions with confidence scores.

---

## 6. Further Research

- **Multi-class Classification:** Extend to more animal classes or breeds.
- **Transfer Learning:** Employ pre-trained models for improved results on smaller datasets.
- **Model Robustness:** Test with adversarial or low-quality images.
- **Mobile Deployment:** Convert and deploy the model to mobile devices using TensorFlow Lite.
- **Explainability:** Integrate Grad-CAM or similar to visualize what parts of the image drive the prediction.

---

## 7. Sample Code Snippets and Explanations

### Data Augmentation

```python
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
```
*Applies random transformations to images for better model generalization.*

### Model Definition

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
*A typical CNN for binary classification.*

### Making a Prediction

```python
img = image.load_img('test.jpg', target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
print('Dog' if prediction[0][0] > 0.5 else 'Cat')
```
*Loads an image, processes it, and predicts the class.*

---

## 8. Deployment & Usage

**Run the Streamlit App:**
1. Clone the repository:
   ```
   git clone https://github.com/roshashaik2326/Ai-Agent-Project.git
   ```
2. Install dependencies:
   ```
   cd dog_cat_classifier
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

**Colab Notebook:**
- Open the provided Colab link.
- Run all cells to train, test, and interact with the classifier.

---

## 9. References

- [TensorFlow Keras Documentation](https://keras.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ImageDataGenerator Guide](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)

---

## 10. Contact & Contributions

- **Author:** [roshashaik2326](https://github.com/roshashaik2326)
- **Feedback/Issues:** Use [GitHub Issues](https://github.com/roshashaik2326/Ai-Agent-Project/issues)
- **Contributions:** Fork, improve, and submit a pull request!

---

> For questions or suggestions about the Dog and Cat Classifier, contact via GitHub. Enjoy experimenting with image deep learning!
