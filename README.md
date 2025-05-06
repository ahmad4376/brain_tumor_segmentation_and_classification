# Brain Tumor Segmentation and Classification

This repository contains the implementation of a deep learning-based solution for the segmentation and classification of brain tumors from MRI scans. The project leverages U-Net with MobileNetV2 for tumor segmentation and EfficientNetB1 for classification, providing an automated tool for medical image analysis. Additionally, a **Streamlit** application has been created to allow real-time analysis by non-technical users.

## Project Overview

Brain tumors are a significant health concern globally, and their early detection is crucial for timely treatment. This project aims to automate the process of tumor segmentation and classification using deep learning techniques, improving both the speed and accuracy of diagnosis. The solution integrates:
1. **Tumor Segmentation** using the **U-Net** architecture utilizing MobileNetV2.
2. **Tumor Classification** using the **EfficientNetB1** architecture.
3. A **Streamlit Application** for easy interaction, allowing users to upload MRI scans, get segmented tumors, and classify them.

## Key Contributions

1. **Segmentation Model**: Implemented using the U-Net architecture for accurate tumor segmentation from MRI images.
2. **Classification Model**: Introduced EfficientNetB1 for classifying segmented tumors into three categories: glioma, meningioma, and pituitary tumor.
3. **Improvement with Transfer Learning**: Enhanced segmentation and classification performance through data augmentation and transfer learning from pre-trained models (MobileNetV2).
4. **Streamlit Application**: Developed an interactive web application that allows users to upload MRI images and receive tumor segmentation and classification results.
5. **Dataset**: The brain tumor dataset used for training is publicly available via [Figshare](https://doi.org/10.6084/m9.figshare.1512427.v8).

## Files in this Repository

- **`brain-tumor-segmentation_baseline.ipynb`**: Jupyter notebook for the baseline model using U-Net for segmentation.
- **`brain-tumor-segmentation_improvement_1.ipynb`**: Jupyter notebook for the first improvement that incorporates data augmentation and transfer learning.
- **`brain-tumor-segmentation_improvement_2.ipynb`**: Jupyter notebook for the second improvement introducing the EfficientNetB1 model for classification.
- **`app.py`**: Streamlit app that allows users to upload MRI images and get tumor segmentation and classification results.
- **`model_best_checkpoint.keras`**: Trained U-Net model for segmentation.
- **`classification_best_checkpoint.keras`**: Trained EfficientNetB1 model for tumor classification.

## Running the Project

### 1. **Run the Notebooks on Kaggle or Google Colab**

You can run the Jupyter notebooks directly on [Kaggle](https://www.kaggle.com/) or [Google Colab](https://colab.research.google.com/). The notebooks are fully set up, and all necessary libraries are installed within the notebooks themselves. You don't need to worry about setting up a `requirements.txt` file.

To run the project:
- Simply upload the respective `.ipynb` file to Kaggle or Colab.
- Run the cells, and the environment will automatically install any additional libraries that are needed.

### 2. **Run the Streamlit App**

To run the **Streamlit** app locally, you can use the following command:

```bash
streamlit run app.py
```

Note: For Streamlit to run successfully, ensure that both the model checkpoint files (model_best_checkpoint.keras for segmentation and classification_best_checkpoint.keras for classification) are in the same directory as the app.py file. These files are required for the app to perform tumor segmentation and classification on the uploaded MRI images.

### 3. **Using Pre-trained Models**
The pre-trained models for segmentation and classification are saved as .keras files and are loaded in the app for inference. Ensure that these models are available in the directory when using the app.



**Acknowledgments**
- The U-Net architecture for segmentation (Ronneberger et al., 2015).

- The EfficientNetB1 model for classification (Tan & Le, 2019).

- The MobileNetV2 encoder for transfer learning (Sandler et al., 2018).

- The Brain Tumor Dataset available on Figshare.
