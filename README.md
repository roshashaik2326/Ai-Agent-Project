# 🐶🐱 Dog and Cat Image Classification

## 1. 📄 Project Content

This project focuses on building a Convolutional Neural Network (CNN) model using deep learning to classify images into two categories: **dogs** and **cats**. The dataset used is a subset of the popular Kaggle Dog vs. Cat dataset. The model is trained using TensorFlow and Keras to learn distinguishing features of dog and cat images.

---

## 2. 🧠 Project Code

### 📦 Imports
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense



# Ai-Agent-Project
## AI-PROJECTS

<table>
  <thead>
    <tr>
      <th>Topic</th>
      <th>PDF Link</th>
      <th>Streamlit App</th>
      <th>Colab Notebook</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Healthcare XAI</td>
      <td><a href="https://yourlink.com/pdf1.pdf"><img src="https://img.shields.io/badge/Open%20in-PDF-red?style=for-the-badge"></a></td>
      <td><a href="https://yourstreamlitapp1.com"><img src="https://img.shields.io/badge/Open%20in-Streamlit-grey?style=for-the-badge&logo=streamlit"></a></td>
      <td><a href="https://colab.research.google.com/yournotebook1"><img src="https://img.shields.io/badge/Open%20in-Colab-blue?style=for-the-badge&logo=googlecolab"></a></td>
    </tr>
    <tr>
      <td>Dog and Cat</td>
      <td><a href="https://yourlink.com/pdf2.pdf"><img src="https://img.shields.io/badge/Open%20in-PDF-red?style=for-the-badge"></a></td>
      <td><a href="https://yourstreamlitapp2.com"><img src="https://img.shields.io/badge/Open%20in-Streamlit-grey?style=for-the-badge&logo=streamlit"></a></td>
      <td><a href="https://colab.research.google.com/yournotebook2"><img src="https://img.shields.io/badge/Open%20in-Colab-blue?style=for-the-badge&logo=googlecolab"></a></td>
    </tr>
    <tr>
      <td>IMDB dataset</td>
      <td><a href="https://yourlink.com/pdf3.pdf"><img src="https://img.shields.io/badge/Open%20in-PDF-red?style=for-the-badge"></a></td>
      <td><a href="https://yourstreamlitapp3.com"><img src="https://img.shields.io/badge/Open%20in-Streamlit-grey?style=for-the-badge&logo=streamlit"></a></td>
      <td><a href="https://colab.research.google.com/yournotebook3"><img src="https://img.shields.io/badge/Open%20in-Colab-blue?style=for-the-badge&logo=googlecolab"></a></td>
    </tr>
  </tbody>
</table>
