### **Instructions**

This folder should contain the dataset files required for training and evaluating the model. Our model uses the **Cityscapes dataset** for semantic segmentation.

#### **Folder Structure**
After downloading and extracting the dataset, the directory should be structured as follows:

```
data/
│-- gtFine/
│   ├── train/
│   ├── val/
│   ├── test/
│-- leftImg8bit/
│   ├── train/
│   ├── val/
│   ├── test/
```

#### **Downloading the Dataset**
To download the Cityscapes dataset, follow these steps:

1. Visit the official dataset page:  
   👉 [Cityscapes Dataset Overview](https://www.cityscapes-dataset.com/dataset-overview/)
2. **Create an account** (required for access).
3. Navigate to the **Download** section.
4. Download the required files for **pixel-level segmentation**.
5. Extract the downloaded files into this `data/` folder, following the structure above.