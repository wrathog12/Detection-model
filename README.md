
```markdown
# 🚀 Object Detection with Faster R-CNN (ResNet-50 + FPN)

This repository contains an end-to-end object detection pipeline using **Faster R-CNN** with a **ResNet-50** backbone and **Feature Pyramid Network (FPN)** for multi-scale feature extraction. The model is trained on the **Pascal VOC 2007** dataset using PyTorch and Google Colab.

---

## 📁 Project Structure

```

├── CNNAssignment.ipynb       # Main training and evaluation notebook
├── README.md                 # Project documentation

````

---

## 🧠 Model Overview

- **Architecture**: Faster R-CNN
- **Backbone**: ResNet-50 (pretrained on ImageNet)
- **Feature Enhancer**: Feature Pyramid Network (FPN)
- **Dataset**: Pascal VOC 2007 (trainval split)
- **Framework**: PyTorch (torchvision.models.detection)

---

## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fasterrcnn-resnet50-fpn.git
   cd fasterrcnn-resnet50-fpn
````

2. Install dependencies (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:

   ```bash
   jupyter notebook CNNAssignment.ipynb
   ```

> 💡 *You can also run it in Google Colab for GPU acceleration.*

---

## 📦 Dataset

This project uses the **Pascal VOC 2007** dataset. Download it using:

```python
from torchvision.datasets import VOCDetection
VOCDetection(root=".", year="2007", image_set="trainval", download=True)
```

---

## 🏋️‍♂️ Training

The notebook includes:

* Dataset loading and preprocessing
* Model setup and transfer learning
* Training loop with optimizer and loss functions
* Visualization of predictions

> Default training is limited to 2 epochs due to runtime constraints, but you can increase `num_epochs` for better accuracy.

---

## 📊 Evaluation

* Visual comparison of predictions and ground truths
* Color-coded bounding boxes
* mAP support can be added optionally

---

## 🤖 AI Tools Used

This project was developed with the assistance of **ChatGPT**, which supported:

* Resolving CUDA and shape errors
* Understanding complex components like RoI Align and anchor generation
* Structuring and editing the final report and README
* Accelerating documentation and code searches

---

## 📌 Note on Training Constraints

> Due to limited GPU access and repeated Colab interruptions, only 2 epochs of training were successfully completed. Multiple attempts failed due to session timeouts and unstable connectivity.

---


## 📄 License

This project is licensed under the MIT License.
