# Mini-HRNet: Lightweight Semantic Segmentation
# Thesis Title: Semantic Image Segmentation using a Modified HRNet

## 📌 Overview
Semantic image segmentation is a fundamental computer vision task that involves classifying each pixel in an image into predefined categories, such as road, sky, or car. This task is critical in various real-world applications, including medical imaging and autonomous driving. High-Resolution Networks (HRNet) are among the most effective architectures for this task. However, HRNet is computationally expensive and requires significant resources to train from scratch.

This project introduces **Mini-HRNet**, a scaled-down version of HRNet that retains modest performance while being more efficient. The model was tested on the **Cityscapes** dataset and achieved **66% mean intersection over union (mIoU) on the validation set** and **64% on the test set**, using only **13% of the trainable parameters** of the original HRNet.

## 🔍 Key Features
- **Lightweight Architecture**: Three different approaches were explored to scale down HRNet while preserving accuracy.
- **Efficient Training**: The model was trained on a Mobile RTX 4070 GPU with limited batch sizes.
- **Custom Data Augmentation**: Normalization, random scaling, cropping, and horizontal flipping were applied to improve generalization.
- **Multi-Scale Inference**: Enhanced model predictions using multi-scale techniques.
- **Flexible Configuration**: Training parameters, loss function weights, and augmentations are fully configurable.

## 🏗️ Model Variants
Three architectural modifications were tested:
1. **Removing the 4th stage** to reduce complexity.
2. **Limiting stage repetitions** to enhance efficiency.
3. **Redesigning the core components**, including replacing the basic block and restructuring the main stages.

Among these, **Approach 3** achieved the best balance of efficiency and performance.

## ⏳ Challenges & Limitations
- **Hardware Constraints**: Training required approximately **30 minutes per epoch**, totaling **10 days per model**.
- **Dataset Imbalance**: Certain classes (e.g., road, sky) dominated the dataset, while rare classes (e.g., truck, pole) were underrepresented.
- **Overfitting Issues**: Despite augmentation techniques, imbalance led to learning biases for major classes.

## 📥 Dataset
The model is trained on the **Cityscapes dataset**. You can download it from:
[Cityscapes Dataset](https://www.cityscapes-dataset.com/dataset-overview/)

Ensure the dataset follows this structure:
```
-- data/
   -- gtFine/
      -- train/
      -- val/
      -- test/
   -- leftImg8bit/
      -- train/
      -- val/
      -- test/
```

## 🛠️ Installation
```bash
git clone https://github.com/alraimi365/mini-hrnet.git
cd mini-hrnet
pip install -r requirements.txt
```

## 🚀 Training the Model
To start training, run:
```bash
python train.py
```
Configurations can be adjusted in `configs/default.yaml`.

## 📜 License
This project is open-source and available under the GPLv3 License.

---

### 📩 Contact
For questions or collaborations, feel free to reach out!
Email: alraimi365@gmail.com
