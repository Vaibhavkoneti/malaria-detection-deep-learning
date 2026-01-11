# 🦟 Malaria Detection Using Deep Learning

This project uses **Deep Learning (CNN)** to classify microscopic blood smear images as **malaria-infected** or **uninfected** cells.  
The dataset is loaded directly from **TensorFlow Datasets (TFDS)**.

---

## 📌 Project Description

Malaria is a life-threatening disease caused by parasites transmitted through mosquito bites. Manual diagnosis is time-consuming and requires expert knowledge. This project automates malaria detection using **Convolutional Neural Networks (CNNs)**, improving speed and accuracy.

---

## 📂 Dataset

- **Dataset Name:** Malaria
- **Source:** TensorFlow Datasets (TFDS)
- **Classes:**
  - Parasitized
  - Uninfected
- **Image Type:** RGB cell images

### Dataset Loading Code
```python
import tensorflow_datasets as tfds

dataset, dataset_info = tfds.load(
    "malaria",
    with_info=True,
    split=[
        'train[:80%]',    # Training data
        'train[80%:90%]', # Validation data
        'train[90%:]'     # Test data
    ]
)
