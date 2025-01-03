from keras import models, layers, callbacks
import matplotlib.pyplot as plt

def compileModel(model, optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def fitModel(model, X_train, y_train, X_test, y_test, epochs, batch_size, patience=3):
    return model.fit(
        X_train, 
        y_train, 
        epochs = epochs, 
        batch_size = batch_size,
        callbacks = callbacks.EarlyStopping(monitor = 'loss', patience = patience),
        validation_data=(X_test, y_test),
        verbose=0)

# no needs, just take the last element of the history
def evaluateModel(model, X_test, y_test):
    return model.evaluate(X_test, y_test, verbose=0)

def saveModel(model, path, modelname):
    model.save(f"{path}/{modelname}'.keras")

def plot_history(history, path, modelname = 'model'):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"{path}/{modelname}_accuracy.png")
    plt.close()

    # Plot training & validation loss values
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"{path}/{modelname}_loss.png")

    plt.close()