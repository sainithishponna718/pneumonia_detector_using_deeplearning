# Pneumonia Detection Using Deep Learning

## Overview

This project investigates the use of deep learning techniques for detecting pneumonia from chest X-ray images. Multiple approaches were explored, including a custom Convolutional Neural Network (CNN), transfer learning using DenseNet121, and fine-tuning experiments.

The objective was to compare training from scratch with transfer learning and evaluate their effectiveness on a medical imaging classification task.

---

## Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)

Classes:

* NORMAL
* PNEUMONIA

Dataset Size:

* Training Images: 5,216
* Test Images: 624

---

## Models Evaluated

### 1. Custom CNN

Architecture:

* 3 Convolutional Layers
* Max Pooling Layers
* Global Average Pooling
* Fully Connected Layer
* Sigmoid Output Layer

#### Results

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 70.99% |
| Test AUC      | 0.684  |

Confusion Matrix:

```text
[[ 67 167]
 [ 14 376]]
```

---

### 2. DenseNet121 (Transfer Learning)

Backbone:

* DenseNet121 pretrained on ImageNet

Classifier Head:

* GlobalAveragePooling2D
* Dense(128, ReLU)
* Dropout(0.3)
* Dense(64, ReLU)
* Dropout(0.3)
* Dense(1, Sigmoid)

Parameters:

* Trainable Parameters: 139,521
* Non-Trainable Parameters: 7,037,504

#### Results

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 84.62% |
| Test AUC      | 0.946  |

Confusion Matrix:

```text
[[145  89]
 [  7 383]]
```

Classification Report:

| Class     | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| Normal    | 0.95      | 0.62   | 0.75     |
| Pneumonia | 0.81      | 0.98   | 0.89     |

---

### 3. DenseNet121 Fine-Tuning

Experiment:

* Unfroze the final layers of DenseNet121
* Fine-tuned using a low learning rate

#### Results

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 81.73% |
| Test AUC      | 0.895  |

Observation:
Fine-tuning improved validation performance but reduced test performance, indicating overfitting on this dataset.

---

## Model Comparison

| Model                         | Test Accuracy | Test AUC  |
| ----------------------------- | ------------- | --------- |
| Custom CNN                    | 70.99%        | 0.684     |
| DenseNet121 (Frozen Backbone) | **84.62%**    | **0.946** |
| DenseNet121 (Fine-Tuned)      | 81.73%        | 0.895     |

---

## Key Findings

* Transfer learning significantly outperformed training a CNN from scratch.
* DenseNet121 achieved the best overall performance.
* Fine-tuning resulted in overfitting and reduced generalization on the test set.
* The final model achieved a pneumonia recall of 98.2%, missing only 7 pneumonia cases out of 390.
* Transfer learning proved highly effective for this relatively small medical imaging dataset.

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn



