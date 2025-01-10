# Image Classification

This repository contains a Python project that implements multple Neural Network Models for image classification using Keras.
The project focuses on classifying images from the MNIST dataset, which consists of handwritten digits.

## Project Structure

- `src/model.py`: Contains the code for building and training the MLP model.
- `src/utils.py`: Contains utility functions for preprocessing data, plotting results, and saving the model.
- `main.py`: Main script to run the model.
- `models/`: Directory to save trained models.
- `results/`: Directory to save plots and model information.

## Dependencies

The project requires the Python libraries listed in the `requirements.txt` file.

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```
## Customization

You can customize the models by modifying the following parameters in `main.py`:

- `num_layers`: Number of hidden layers in the MLP.
- `layers_units`: Number of units in each hidden layer.
- `activation`: Activation function for the hidden layers.
- `optimizer`: Optimizer used for training the model.
- `epochs`: Number of training epochs.
- `batch_size`: Batch size for training.

## Results

The project saves the trained model in the `models/` directory, plots of weights distribution, accuracy and loss in the `results/plots/` directory, and a csv file containing model information in the `results/` directory.

