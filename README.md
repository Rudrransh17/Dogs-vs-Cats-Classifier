# Dogs vs Cats Classifier

This is a Dogs vs Cats Classifier that utilizes transfer learning to classify images as either dogs or cats. The classifier was trained using a dataset downloaded from the Kaggle competition [Dogs vs Cats](https://www.kaggle.com/competitions/dogs-vs-cats).

## Dataset

The dataset contains a total of 12,500 images, split equally between cats and dogs. To prepare the dataset for training, it was organized into directories following a specific format that allows the `ImageDataGenerator` to load the images properly. The dataset was further divided into a training set and a validation set using a 0.2 ratio for the validation images.

## Image Loading and Augmentation

The images were loaded and augmented using the `ImageDataGenerator` class from TensorFlow. Data augmentation is a technique that generates additional training samples by applying various transformations to the existing images. This augmentation process helps the model generalize better and reduces overfitting.

During training, the images were augmented using the following techniques:
- Rotation: Randomly rotates the image within a range of 45 degrees.
- Width Shift: Shifts the image horizontally by a fraction of its width.
- Height Shift: Shifts the image vertically by a fraction of its height.
- Shear Range: Applies shearing transformations to the image.
- Zoom Range: Applies random zooming to the image.
- Horizontal Flip: Flips the image horizontally.

The validation images were only rescaled, without any augmentation, to ensure fair evaluation during training.

## Model Initialization

For transfer learning, the pre-trained InceptionV3 model was used. The weights of the InceptionV3 model were downloaded from the following URL: [InceptionV3 Model Weights](https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5).

The InceptionV3 model was loaded without the top layers, as they were not necessary for the specific Dogs vs Cats classification task. By excluding the top layers, we can add our custom dense network on top of the InceptionV3 base model.

## Adding Dense Network

A custom dense network was added on top of the pre-trained InceptionV3 model to perform specific training on the Dogs vs Cats dataset. The dense network consists of the following layers:
- Flatten: Converts the output of the previous layer into a 1-dimensional vector.
- Dense (with ReLU activation): A fully connected layer with 1024 neurons, providing non-linearity to the model.
- Dropout: Helps prevent overfitting by randomly setting a fraction of input units to 0 during training.
- Dense (with Sigmoid activation): The final layer with a single neuron, using a sigmoid activation function for binary classification (cat or dog).

The dense network was appended to the InceptionV3 base model, creating a new model architecture.

## Training

The model was compiled using the Adam optimizer, binary cross-entropy loss function, and accuracy as the evaluation metric. During training, the augmented training images were fed to the model in batches of 100, while the validation images were fed in batches of 25.

The model was trained for 20 epochs, and the training progress was logged. The training and validation accuracy were recorded for each epoch.

## Model and Plot

The trained model was saved; however, due to its large size (>100MB), it is not included in this repository. Instead, a plot of the validation and training accuracy is provided in the repository as `plot.jpg`. You can refer to this image to visualize the training progress and the model's performance.    
![alt text](https://raw.githubusercontent.com/Rudrransh17/Dogs-vs-Cats-Classifier/main/plot.jpg)

## Conclusion

The Dogs vs Cats Classifier demonstrates the use of transfer learning with the InceptionV3 model to classify images of cats and dogs. By leveraging a pre-trained model and fine-tuning it with a custom dense network, the classifier achieves reasonable accuracy in distinguishing between the two classes. This classifier serves as a starting point for further exploration and improvement in image classification tasks involving dogs and cats.
