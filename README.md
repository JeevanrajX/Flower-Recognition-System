\# Flower Recognition System

\## Deep Learning-Based Image Classification Using MobileNetV2



\### 1. Project Overview



The Flower Recognition System is a deep-learning-based image classification application developed using Python and PyTorch.



It uses a pretrained MobileNetV2 convolutional neural network with transfer learning to classify uploaded images into five flower categories:



\- Daisy

\- Dandelion

\- Roses

\- Sunflowers

\- Tulips



The application provides a web-based interface developed using Flask. Users can upload an image and view the predicted flower category along with class-wise confidence scores.



\### 2. Performance



| Evaluation Metric | Result |

|---|---:|

| Dataset size | 3,670 images |

| Training images | 2,569 |

| Validation images | 550 |

| Testing images | 551 |

| Best validation accuracy | 92.18% |

| Test accuracy | 93.10% |

| Correct test predictions | 513 / 551 |



\### 3. Technology Stack



\- Python 3.10.11

\- PyTorch

\- Torchvision

\- MobileNetV2

\- Flask

\- OpenCV

\- NumPy

\- Pillow

\- Scikit-learn

\- Matplotlib



\### 4. Project Structure



```text

ImageRecognition\_Submission/

│

├── app.py

├── requirements.txt

├── README.md

│

├── src/

│   ├── train.py

│   ├── evaluate.py

│   └── analyze\_errors.py

│

├── models/

│   ├── flower\_classifier.pth

│   ├── evaluation\_report.txt

│   └── confusion\_matrix.png

│

├── templates/

│   └── index.html

│

├── static/

│   └── style.css

│

└── dataset\_metadata.json

```



Note: The filenames shown under `src/` should be adjusted to match the actual files included in the submission.



\### 5. System Requirements



Recommended environment:



\- Windows 10 or Windows 11

\- Python 3.10.x

\- Minimum 8 GB RAM; 16 GB recommended

\- NVIDIA GPU with compatible CUDA support (optional)



The trained model can run on a CPU if the application supports CPU device selection. An NVIDIA GPU is not mandatory for inference.



\### 6. Installation Instructions



\*\*Step 1 – Install Python\*\*



Install Python 3.10 from:



https://www.python.org/downloads/



Verify installation:



```powershell

python --version

```



\*\*Step 2 – Extract the project\*\*



Extract the submitted ZIP archive into a suitable directory.



Example:



```text

D:\\ImageRecognition\_Submission

```



\*\*Step 3 – Open PowerShell in the project directory\*\*



```powershell

cd D:\\ImageRecognition\_Submission

```



\*\*Step 4 – Create a virtual environment\*\*



```powershell

python -m venv .venv

```



\*\*Step 5 – Activate the environment\*\*



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



If PowerShell blocks activation, run:



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

```



Then activate the environment again.



\*\*Step 6 – Install dependencies\*\*



```powershell

python -m pip install --upgrade pip

pip install -r requirements.txt

```



Important: The requirements file must contain compatible package versions for the target computer. If the submitted environment uses CUDA-specific PyTorch packages, CPU-only or different GPU systems may require an adjusted PyTorch installation.



\### 7. Running the Application



Ensure that the trained model exists at:



```text

models/flower\_classifier.pth

```



Start the Flask application:



```powershell

python app.py

```



Open the following address in a web browser:



```text

http://127.0.0.1:5000

```



\### 8. Using the Application



1\. Open the application in your browser.

2\. Select an image from your computer.

3\. Preview the selected image.

4\. Submit the image for classification.

5\. Wait for the prediction.

6\. View the predicted flower category and confidence scores.



\### 9. Dataset Information



The project uses the TensorFlow Flowers dataset.



Dataset source:



https://www.tensorflow.org/datasets/catalog/tf\_flowers



The original dataset is not included in the compact submission package to reduce archive size.



The trained model checkpoint is included, so the Flask application should not require the original training dataset for normal inference.



The dataset is required if the recipient intends to retrain the model or reproduce the complete evaluation workflow.



\### 10. Model Information



Architecture: MobileNetV2



Training approach: Transfer Learning and Partial Fine-Tuning



Input resolution: 224 × 224 pixels



Number of output classes: 5



Optimizer: AdamW



Loss function: Cross-Entropy Loss



Batch size: 32



Training schedule:

\- 5 epochs of classifier warm-up.

\- 10 epochs of partial fine-tuning.



\### 11. Evaluation Artifacts



The submission includes:



\- `models/evaluation\_report.txt` – Classification metrics.

\- `models/confusion\_matrix.png` – Confusion matrix visualization.

\- `models/flower\_classifier.pth` – Trained model checkpoint.



\### 12. Important Notes and Limitations



\- The model is designed to classify five flower categories.

\- It is not a general-purpose object recognition model.

\- Non-flower images may still be assigned one of the five flower labels.

\- Confidence scores should not be interpreted as guaranteed probabilities that an image is a flower.

\- GPU acceleration is optional for inference, subject to the application's device configuration.



\### 13. Contact and Project Information



Project: Flower Recognition System



Developed by: Jeevanraj M



Programme: B.Tech – Computer Science and Engineering



Institution: Sri Manakula Vinayagar Engineering College



Internship Task: Image Recognition System

