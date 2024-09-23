# Flower-Classifier: AI-Powered Flower Recognition

<p align="center">
<!-- <img src="https://imgur.com/5wSdLCq.png" width="500"> -->
<img src="https://imgur.com/u5OUfBF.png" width="500">
</p>

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Command Line Application](#command-line-application)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Performance](#performance)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Project Overview

The Flower-Classifier project is an advanced image classification system developed as part of the AWS AI/ML Scholarship program at Udacity. This project demonstrates the application of deep learning techniques to accurately classify images of flowers into their respective species.

Using state-of-the-art convolutional neural networks and transfer learning, this classifier achieves high accuracy in identifying various flower species. The project showcases the entire machine learning pipeline, from data preprocessing to model deployment, and provides both a Jupyter notebook for detailed analysis and a command-line interface for easy use.

## Features

- High-accuracy flower species classification
- Utilizes transfer learning with pre-trained models
- Comprehensive Jupyter notebook detailing the development process
- Command-line interface for training and making predictions
- Supports custom datasets and model architectures
- Implements best practices in AI/ML project structure and documentation

## Repository Structure

```
Flower-Classifier/
│
├── assets/                  # Project assets and images
├── flowers/                 # Flower dataset directory
├── .gitignore               # Git ignore file
├── 00_Image Classifier Project.ipynb    # Main Jupyter notebook
├── 01_Image Classifier Project.ipynb     # Main Jupyter notebook with Output
├── Image Classifier Project.html        # HTML export of the notebook
├── CODEOWNERS               # Defines code ownership
├── LICENSE                  # MIT License file
├── README.md                # This file
├── cat_to_name.json         # Mapping of categories to flower names
├── predict.py               # Script for making predictions
└── train.py                 # Script for training the model
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Sourabh-Kumar04/Flower-Classifier.git
   cd Flower-Classifier
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Jupyter Notebook

1. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open `01_Image Classifier Project.ipynb`.

3. Ensure you have a GPU runtime enabled for faster training.

4. Follow the instructions in the notebook to load data, build, train, and evaluate the model.

### Command Line Application

1. Train the model:
   ```
   python train.py --data_dir flowers --learning_rate 0.001 --hidden_units 512 --epochs 5
   ``` 
   for GPU :
   ```
   python train.py flowers --save_dir checkpoint.pth --arch vgg16 --epochs 5 --print_every 20 --lr 0.001 --dropout 0.3 --hidden_layers 256 --gpu gpu
   ```
   for CPU :
   ```
   python train.py flowers --save_dir checkpoint.pth --arch vgg16 --epochs 5 --print_every 20 --lr 0.001 --dropout 0.3 --hidden_layers 256 --gpu cpu
   ```

2. Make predictions:
   ```
   python predict.py path/to/image checkpoint.pth --top_k 5 --category_names cat_to_name.json
   ```
   or
   ```
   python predict.py input_img --checkpoint 'checkpoint.pth' --top_k 5 --cat_to_name 'cat_to_name.json' --gpu cuda
   ```

   Replace `path/to/image` with the path to your flower image and `checkpoint.pth` with your saved model checkpoint.

## Model Architecture

The project uses a pre-trained convolutional neural network (e.g., VGG16 or ResNet) as a feature extractor, followed by a custom classifier. The exact architecture can be customized in the `train.py` script.

## Dataset

The model is trained on a dataset of 102 flower categories. Each class contains between 40 and 258 images. The dataset is split into training, validation, and testing sets.

## Performance

The model achieves an accuracy of over 80% on the test set. Detailed performance metrics and confusion matrices can be found in the Jupyter notebook.

## Technologies Used

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

## Future Improvements

- Implement data augmentation for improved generalization
- Experiment with more advanced architectures (e.g., EfficientNet)
- Develop a web interface for easy online predictions
- Expand the dataset to include more flower species

## Contributing

Contributions to improve the Flower-Classifier project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Udacity for providing the project structure and guidance
- AWS for sponsoring the AI/ML Scholarship program
- The PyTorch team for their excellent deep learning framework

## Contact

Sourabh Kumar - [GitHub Profile](https://github.com/Sourabh-Kumar04)

Project Link: [Flower-Classifier](https://github.com/Sourabh-Kumar04/Flower-Classifier)

For any additional questions or comments, please open an issue in the GitHub repository.