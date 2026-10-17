# CIFAR-10 Image Classification with PyTorch

A deep learning project exploring Convolutional Neural Networks (CNNs), data augmentation, transfer learning, and fine-tuning using PyTorch and the CIFAR-10 dataset.

The project compares custom CNN architectures with ResNet18 and evaluates their performance on a 10-class image classification task.

## Project Results

The best-performing model in the recorded experiments was **ResNet18 Fine-Tuning**, achieving **93.41% test accuracy** on CIFAR-10.

| Model | Test Accuracy |
|---|---:|
| SimpleCNN | 71.50% |
| SimpleCNN + Data Augmentation | 75.51% |
| ImprovedCNN + Data Augmentation | 82.54% |
| ImprovedCNN + Learning Rate Scheduler | 88.53% |
| ResNet18 Feature Extraction | 86.49% |
| **ResNet18 Fine-Tuning** | **93.41%** |

*These results come from the recorded experiments. Training configurations differ, so this table should be considered a comparison of experimental runs rather than a controlled benchmark.*

## Dataset

This project uses [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), a dataset of 60,000 color images across 10 classes.

- 50,000 training images
- 10,000 test images
- Image dimensions: 32 × 32 pixels
- Number of classes: 10

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The original training set was divided into 45,000 training images and 5,000 validation images.

The dataset is downloaded automatically by the training and evaluation data pipeline when needed.

## Methods

### 1. Custom CNN Architectures

Implemented and compared two custom convolutional neural networks: `SimpleCNN` and `ImprovedCNN`.

### 2. Data Augmentation

Applied random cropping and random horizontal flipping to training images to improve generalization.

### 3. Learning Rate Scheduling

Used `ReduceLROnPlateau` to reduce the learning rate when validation performance stopped improving.

### 4. Transfer Learning

Used pretrained ResNet18 weights and replaced its final classification layer with a layer for the 10 CIFAR-10 classes.

### 5. Fine-Tuning

Unfroze the ResNet18 `layer4` block and final fully connected layer. Separate learning rates were used for these layers.

### 6. Model Evaluation

Evaluated models using test accuracy, test loss, per-class precision, recall, F1-score, and confusion matrices.

## Final Model Evaluation

| Metric | Result |
|---|---:|
| Test Accuracy | 93.41% |
| Test Loss | 0.2343 |
| Test Images | 10,000 |

### Confusion Matrix

![ResNet18 Fine-Tuning Confusion Matrix](outputs/confusion_matrix_resnet18_finetune.png)

The model performs well across most classes. Cat and dog are among the more challenging classes, with misclassifications occurring in both directions.

The classification report provides additional per-class evaluation metrics.

## Pretrained Model

The fine-tuned ResNet18 checkpoint is available on Hugging Face.

- **Model:** [CIFAR-10 ResNet18 Fine-Tuned](https://huggingface.co/aidinpho/cifar10-resnet18-finetuned)
- **Test accuracy:** 93.41%
- **Test loss:** 0.2343
- **Checkpoint:** `best_resnet18_finetune.pth`
- **Checkpoint size:** Approximately 44.8 MB

Model weights are hosted separately to keep the GitHub repository lightweight.

Download `best_resnet18_finetune.pth` from the Hugging Face repository and place it in the project's `models/` directory before running evaluation or prediction.

## Interactive Prediction Demo

The project includes an interactive script that lets you select an image from your computer and view the model's top three predictions.

After downloading the checkpoint, run:

```bash
python src/predict.py
```

The script opens a file-selection dialog and displays the top three predicted classes with their model scores.

**Note:** This demo is intended for images resembling the CIFAR-10 classes. Its results on arbitrary internet images may differ from its benchmark performance on the CIFAR-10 test set. The displayed scores are softmax outputs and are not necessarily calibrated confidence estimates.

## Project Structure

```text
image-classification/
├── outputs/                 # Selected evaluation results and plots
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   ├── train_improved.py
│   ├── train_improved_v2.py
│   ├── train_resnet18.py
│   ├── train_resnet18_finetune.py
│   ├── evaluate.py
│   ├── evaluate_improved_v2.py
│   ├── evaluate_resnet18_finetune.py
│   ├── compare_models.py
│   ├── explore_dataset.py
│   ├── plot_history.py
│   ├── plot_improved_history.py
│   ├── visualize_batch.py
│   ├── visualize_errors.py
│   ├── visualize_errors_v2.py
│   ├── test_dataloader.py
│   ├── test_improved_model.py
│   ├── test_optimizer_step.py
│   └── test_training_step.py
├── test/                    # Manual prediction images
├── tests/
│   └── test_model.py        # Automated pytest tests
├── data/                    # Downloaded dataset (not tracked)
├── models/                  # Model checkpoints (not tracked)
├── .gitignore
├── requirements.txt
└── README.md
```

The `data/` and `models/` directories are excluded from Git. The structure above lists the main project files; generated files and Python cache directories may be omitted.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aidinpho/image-classification.git
cd image-classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The project uses PyTorch and Torchvision. A compatible Python and PyTorch installation is required; GPU acceleration is optional.

## Running the Project

### Train the fine-tuned ResNet18 model

```bash
python src/train_resnet18_finetune.py
```

The training pipeline downloads CIFAR-10 when needed, trains the model, and saves the best checkpoint based on validation performance.

### Evaluate the trained model

First, download the checkpoint from the [Hugging Face model repository](https://huggingface.co/aidinpho/cifar10-resnet18-finetuned) and place it at:

```text
models/best_resnet18_finetune.pth
```

Then run:

```bash
python src/evaluate_resnet18_finetune.py
```

The evaluation script uses the trained checkpoint and generates a classification report, confusion matrix, and JSON summary under `outputs/`.

### Run interactive image prediction

After placing the checkpoint in the `models/` directory, run:

```bash
python src/predict.py
```

Select an image in the file dialog to view the model's top three predictions.

## Automated Tests

Run the automated tests with:

```bash
python -m pytest tests -v
```

The current tests check output shapes for the custom CNNs and ResNet18, verify that softmax probabilities sum to one, and check the expected number of CIFAR-10 classes.

These are primarily architecture and smoke tests; they do not independently validate the accuracy of the trained checkpoint.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- scikit-learn
- pandas
- pytest

## Key Learnings

- Building and training CNN architectures
- Applying data augmentation
- Monitoring validation performance
- Using learning rate schedulers
- Applying transfer learning and fine-tuning
- Evaluating classification models with multiple metrics
- Analyzing confusion matrices and classification errors
- Writing automated tests for machine learning code

## Future Improvements

- Compare additional architectures and optimizers under consistent training configurations
- Add more comprehensive automated tests
- Analyze misclassified images
- Improve per-class performance
- Explore object detection with architectures such as YOLO

## Author

Developed as a hands-on deep learning project to explore image classification, transfer learning, model evaluation, and PyTorch.
