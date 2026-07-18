# Fish Classification Project

This repository contains the source code for a Fish Classification project using Machine Learning/Deep Learning. 

## Dataset

Because the dataset and model weights are extremely large, they are not included directly in this repository.

To run this project locally, you will need to download the dataset and place it in the appropriate folder:

1. Download **A Large Scale Fish Dataset** from Kaggle:
   [https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)
2. Extract the downloaded dataset.
3. Place the extracted files inside the `data/` folder in this repository.

## How to Run

1. Make sure you have installed all the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
2. After the dataset is properly placed in the `data/` folder, you can run the application (e.g., `app.py`).

## Notes
- The `models/` directory is also ignored from this repository to prevent uploading large `.keras` / `.h5` files.
- You can retrain the model locally to generate the files inside the `models/` directory.
