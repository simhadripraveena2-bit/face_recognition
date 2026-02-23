# **🧠 Face Embedding System**

## **🔍 Overview**

This project implements a full-stack face embedding and identification system using deep learning and web deployment.

It uses the VGGFace2 dataset to train a face embedding model, capable of identifying individuals from images. The system includes:
* ML Pipeline: Preprocessing, embedding extraction, classifier training, evaluation.
* Web Application: FastAPI backend + Django frontend for real-time image uploads and identification.

The integrated setup allows a user to upload an image, and the system will predict the person’s identity if present in the database

## **🏗️ Project Structure**

```
missing_person_detection/
├── data/
│   └── vggface2/                     # VGGFace2 images organized by person/
├── models/
│   └── classifier_checkpoint.pth     # Trained classifier checkpoint
├── src/
│   ├── preprocessing.py              # Detect & crop faces
│   ├── feature_extraction.py         # Generate embeddings with VGGFace2
│   ├── datasets.py                   # PyTorch Dataset class
│   ├── models.py                     # MLP classifier architecture
│   ├── training.py                   # Training loop & checkpoint saving
│   ├── evaluation.py                 # Accuracy
│   ├── utils.py                      # Helper functions
│   ├── main.py                       # Full ML pipeline
│   └── inference.py                  # Model wrapper for serving
│
├── app_backend/
│   ├── main.py                       # FastAPI backend
│   ├── requirements.txt              # Backend-specific dependencies
│   └── model_store/                  # Optional folder for uploaded images
│
├── app_frontend/
│   ├── manage.py
│   ├── requirements.txt              # Frontend dependencies
│   ├── frontend_project/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   └── webapp/
│       ├── __init__.py
│       ├── views.py
│       ├── urls.py
│       ├── templates/
│       │   └── webapp/
│       │       └── index.html
│       └── static/
│
├── requirements.txt                  # Optional global dependencies
└── README.md
```

## **📦 Dataset Description**
### VGGFace2 Dataset:
[Kaggle Link](https://www.kaggle.com/datasets/hearfool/vggface2)
* ~3.3 million images of 9,000+ individuals.
* Each folder corresponds to one person with multiple images across pose, age, illumination, and expression.
* Used to train a robust face embedding model, forming the foundation for missing-person identification.

Note: For real-world deployment, this can be replaced with a specialized person dataset.

## **⚙️ How It Works**

* Preprocessing (src/preprocessing.py)
  * Detects faces using MTCNN. 
  * Crops and resizes them to uniform dimensions.
* Feature Extraction (src/feature_extraction.py)
  * Converts cropped faces into 512-dimensional embeddings using pretrained VGGFace2 models.
* Dataset Loading (src/datasets.py)
  * Organizes embeddings and labels into PyTorch Dataset objects. 
  * Supports train/validation/test splits.
* Model Training (src/training.py)
  * Defines a lightweight MLP classifier. 
  * Uses cross-entropy loss and Adam optimizer. 
  * Saves best model as models/classifier_checkpoint.pth.
* Evaluation (src/evaluation.py)
  * Computes accuracy, and confusion matrix for performance analysis.
* Inference (src/inference.py)
  * Loads the trained model checkpoint. 
  * Accepts new images and predicts identity.
* Backend (FastAPI) (app_backend/main.py)
  * Accepts image uploads via HTTP POST. 
  * Runs inference and returns predicted identity + confidence score.
* Frontend (Django) (app_frontend/webapp/)
  * Provides a simple web interface for uploading images. 
  * Displays predicted identity returned by the backend.
* Main Controller (src/main.py)
  * Runs the full ML pipeline: preprocessing → embedding extraction → training → evaluation → launches backend API.

## **🧮 Training & Running the System**

#### Install dependencies
```commandline
pip install -r requirements.txt
```
#### Run the full pipeline
```commandline
python src/main.py
```
* Trains the model.
* Saves classifier_checkpoint.pth

#### launch web app
Run backend (FastAPI)
  * From repo root, activate venv and install backend deps:
  ```commandline
    pip install -r app_backend/requirements.txt
  ```
  * Start backend:
  ```commandline
  uvicorn app_backend.main:app --reload --host 0.0.0.0 --port 8000
  ```
* Run frontend (Django)
  * In a separate venv, install:
  ```commandline
  pip install -r app_frontend/requirements.txt
  ```
  * Start Django devserver:
  ```commandline
  cd app_frontend
  python manage.py runserver 8001
  ```
#### Upload an image via the Django frontend 
* Open http://127.0.0.1:8001/ and upload an image — backend should be running at http://127.0.0.1:8000/.
* The system predicts the identity of the person in the image.
* Returns the name

## **📊 Evaluation Metrics**
* Accuracy
* Classification Report
* Confusion Matrix

## **🧑‍🔬 Research & Impact**
* Demonstrates expertise in deep learning for facial embeddings and recognition.
* Implements end-to-end face embedding and recognition with production-ready deployment.
* Full-stack integration shows ability to bridge ML research and practical application.
