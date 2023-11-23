# NER Mountain Names Task

## Overview

This project addresses the Named Entity Recognition (NER) task specifically focused on identifying mountain names in text. The solution employs a simple RNN-based model trained on a dataset created for this purpose. 
Most of the solution explanation you can find in `mountain-name-recognition.ipynb` which is basically the main code of this solution.
Both `training_process.py` and `inference_process.py` Python scripts are mainly snippets taken from `mountain-name-recognition.ipynb` for running model training and inference on the dataset.

## Directory Structure

```
├── data
│   ├── (your dataset files)
├── saved_weights
│   ├── (saved model weights)
├── Dataset_creation.ipynb
├── inference_process.py
├── mountain-name-recognition.ipynb
├── requirements.txt
├── training_process.py
├── README.md
```

## Files Description

- **Dataset_creation.ipynb:** Jupyter notebook for creating and preprocessing the dataset.
- **inference_process.py:** Python script for running model inference on new data.
- **mountain-name-recognition.ipynb:** Main demo code showcasing the entire process from dataset creation to model training and inference.
- **requirements.txt:** File specifying project dependencies.
- **training_process.py:** Python script for training the NER model.

## How to Set Up the Project

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/your-repository.git
    cd your-repository
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Data Preparation:**
    - Place your dataset files in the `data` directory.

4. **Training the Model:**
    - Execute `training_process.py` to train the NER model.

5. **Inference:**
    - Use `inference_process.py` to run model inference on new data.

6. **Demo:**
    - Explore the complete process using `mountain-name-recognition.ipynb` for a step-by-step demonstration.

## Important Notes

- Ensure that you have the necessary dependencies installed as specified in `requirements.txt`.
- Adjust file paths and configurations in the scripts as needed for your specific setup.
- Make sure to have the required datasets in the `data` directory.
