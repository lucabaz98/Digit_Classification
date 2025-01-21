from keras import models, layers, callbacks
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json

main_dir =  os.path.join(os.path.dirname(os.getcwd()))


def compile_model(model, optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]):
    """Compiles a Keras model with specified optimizer, loss, and metrics."""
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def fit_model(model, X_train, y_train, X_test, y_test, epochs, batch_size, patience=3):
    """Trains a Keras model"""
    return model.fit(
        X_train, 
        y_train, 
        epochs = epochs, 
        batch_size = batch_size,
        callbacks = callbacks.EarlyStopping(monitor = 'loss', patience = patience),
        validation_data=(X_test, y_test),
        verbose=0)

# no needs, just take the last element of the history
def evaluate_model(model, X_test, y_test):
    """Evaluates a Keras model on the test data and returns the results."""
    return model.evaluate(X_test, y_test, verbose=0)

def save_model(model, path, modelname):
    """Saves a Keras model to the specified filepath."""
    model.save(f"{path}/{modelname}.keras")

def plot_accuracy(history, filepath):
    """Plots the training and validation accuracy curves and saves the figure."""
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(filepath)
    plt.close()
    
def plot_loss(history, filepath):
    """Plots the training and validation loss curves and saves the figure."""
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(filepath)
    plt.close()

def one_hot_encoding(y, num_classes):
    """Performs one-hot encoding on categorical labels."""
    return np.eye(num_classes)[y]


def preprocessing(X):
    """Normalizes and flattens input data."""
    X = X.astype(np.float32) / 255.0
    X = X.reshape(X.shape[0], -1)
    return X
   
def results_table(model, modelname, optimizer, epochs, batch_size, activation, regularizer, regularizer_value, initializer, history):
    layers_units = []
    for layer in model.layers[1:]:
        if not isinstance(layer, layers.BatchNormalization):
            layers_units.append(layer.units)
        
    model_info = {
            'model_name': modelname,
            'accuracy': round(history.history['accuracy'][-1],4),
            'val_accuracy': round(history.history['val_accuracy'][-1],4),
            'loss': round(history.history['loss'][-1],4),
            'val_loss': round(history.history['val_loss'][-1],4),
            'num_layers': len(model.layers),
            'layers_units': layers_units,
            'real_epochs': len(history.history['accuracy']),
            'epochs': epochs,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'activation': activation,
            'Regularizer': regularizer,
            'Regularizer_value': regularizer_value,
            'Initializer': initializer
        }
    return model_info
    
    
def save_results(model_info, filepath):    
    # Load existing JSON or create a new one
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        data.append(model_info)  # Add new model info
    else:
        data = [model_info]  # Create a list with the first model info

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)  # Save with indentation for readability
    
    pd.DataFrame(data).to_csv(f'{main_dir}/results/models_info.csv', index=False)    

def plot_weights(model, filepath):
    """
    Plots histograms of the weights for layers in a Keras model.
    """
    # Get indices of all hidden layers
    hidden_layer_indices = list(range(1, len(model.layers)))  
    num_layers = len(hidden_layer_indices)  # Number of hidden layers
    
    fig, axes = plt.subplots(1, num_layers, figsize=(12, 4)) 

    for i, layer_index in enumerate(hidden_layer_indices):
        weights = model.layers[layer_index].get_weights()[0].flatten()  # Get weights and flatten
        axes[i].hist(weights, bins=50) 
        axes[i].set_title(f'{i+1}-layer')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# Running a Keras model, saving it, plotting results, and saving model information.
def run_model(model, modelname, X_train, y_train, X_test, y_test, optimizer, epochs, batch_size, activation, regularizer, regularizer_value, initializer):
    compile_model(model, optimizer=optimizer)

    history = fit_model(model, X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
    
    save_model(model, f'{main_dir}/models', modelname)

    plot_weights(model, f'{main_dir}/results/plots/{modelname}_weights.png')
    plot_accuracy(history, f'{main_dir}/results/plots/{modelname}_accuracy.png')
    plot_loss(history, f'{main_dir}/results/plots/{modelname}_loss.png')
   
    model_info = results_table(model, modelname, optimizer, epochs, batch_size, activation, regularizer, regularizer_value, initializer, history)
    save_results(model_info, f'{main_dir}/results/models_info.json')
    
    
 