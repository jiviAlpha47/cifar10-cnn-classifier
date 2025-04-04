🧠 CIFAR-10 Image Classifier
This project is a Convolutional Neural Network (CNN) built using PyTorch to classify images from the CIFAR-10 dataset. The model is trained to recognize 10 classes of objects including planes, cars, birds, and more.

📂 Dataset
The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. It's divided into 50,000 training images and 10,000 test images.

Classes:
airplane automobile bird cat deer dog frog horse ship truck

🏗️ Model Architecture
The CNN model consists of:
2 Convolutional layers with ReLU and MaxPooling
3 Fully connected layers
Cross-entropy loss with SGD optimizer

Optimizer: SGD with momentum
Learning rate: 0.001
Loss function: CrossEntropyLoss
Epochs: 5 (can be changed)
Accuracy and loss printed per batch and epoch

📊 Evaluation
Accuracy calculated on the test set
Confusion Matrix plotted using sklearn
Inference supported for custom images

🖼️ Custom Inference
To test the model on your own images:

Save your image(s) in the working directory
Update the image_paths list in the code
Images are automatically resized and normalized

📈 Sample Output
Final Test Accuracy
Confusion Matrix plot
Predictions displayed on custom images

📎 Credits
Built with PyTorch & torchvision
Dataset: CIFAR-10 by the University of Toronto
