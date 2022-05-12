# Tensorflow / Keras
from statistics import mode
import plotly
import plotly.graph_objects as go
import plotly.express as px
import Toolkit_dev as myTK
from sklearn.metrics import classification_report  # for model evaluation metrics
from sklearn.model_selection import train_test_split  # for splitting data into train and test samples
import sklearn  # for model evaluation
import numpy as np  # for data manipulation
import pandas as pd  # for data manipulation
from keras.layers import Dense  # for creating regular densely-connected NN layers.
from keras import Input  # for instantiating a keras tensor
from keras.models import Sequential  # for creating a linear stack of layers for our Neural Network
from tensorflow import keras  # for building Neural Networks



def main() -> None:
    myTK.clear()
    
    pd.set_option('display.max_columns', 50)
    
    AUS_Rain_Record = pd.read_csv("weatherAUS.csv", encoding="utf-8")
    AUS_Rain_Record = AUS_Rain_Record[pd.isnull(AUS_Rain_Record["RainTomorrow"])==False]
    AUS_Rain_Record = AUS_Rain_Record.fillna(AUS_Rain_Record.mean(numeric_only=True))

    AUS_Rain_Record["RainTodayFlag"] = AUS_Rain_Record["RainToday"].apply(lambda x: 1 if x == "Yes" else 0)
    AUS_Rain_Record["RainTomorrowFlag"] = AUS_Rain_Record["RainTomorrow"].apply(lambda x: 1 if x == "Yes" else 0)

    # select modeling data
    X = AUS_Rain_Record[["Humidity3pm"]]
    y = AUS_Rain_Record["RainTomorrowFlag"].values
    
    # create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # specify NN shape
    model = Sequential(name="Model-with-One-Input")
    model.add(Input(shape=(1,), name="Input-Layer"))
    model.add(Dense(2, activation='softplus', name='Hidden-Layer'))
    model.add(Dense(1, activation='sigmoid', name='Output-Layer'))
    
    # compile model
    model.compile(  optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["Accuracy", "Precision", "Recall"],
                    loss_weights=None,
                    weighted_metrics=None,  # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                    run_eagerly=None,  # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                    steps_per_execution=None  # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                    )
    
    #fit keras model on the dataset
    model.fit(X_train,  # input data
              y_train,  # target data
              batch_size=10,  # Number of samples per gradient update. If unspecified, batch_size will default to 32.
              epochs=3,  # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
              verbose='auto',  # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
              callbacks=None,  # default=None, list of callbacks to apply during training. See tf.keras.callbacks
              validation_split=0.2,  # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
              #validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
              shuffle=True,  # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
              class_weight={0: 0.3, 1: 0.7},  # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
              sample_weight=None,  # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
              initial_epoch=0,  # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
              steps_per_epoch=None,  # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
              validation_steps=None,  # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
              validation_batch_size=None,  # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
              validation_freq=3,  # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
              max_queue_size=10,  # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
              workers=12,  # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
              use_multiprocessing=False,  # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
              )
    

    ##### Step 6 - Use model to make predictions
    # Predict class labels on training data
    pred_labels_tr = (model.predict(X_train) > 0.5).astype(int)
    # Predict class labels on a test data
    pred_labels_te = (model.predict(X_test) > 0.5).astype(int)


    ##### Step 7 - Model Performance Summary
    print("")
    print('-------------------- Model Summary --------------------')
    model.summary()  # print model summary
    print("")
    print('-------------------- Weights and Biases --------------------')
    for layer in model.layers:
        print("Layer: ", layer.name)  # print layer name
        print("  --Kernels (Weights): ", layer.get_weights()[0])  # weights
        print("  --Biases: ", layer.get_weights()[1])  # biases

    print("")
    print('---------- Evaluation on Training Data ----------')
    print(classification_report(y_train, pred_labels_tr))
    print("")

    print('---------- Evaluation on Test Data ----------')
    print(classification_report(y_test, pred_labels_te))
    print("")
    
    # Create 100 evenly spaced points from smallest X to largest X
    X_range = np.linspace(X.min(), X.max(), 100)
    # Predict probabilities for rain tomorrow
    y_predicted = model.predict(X_range.reshape(-1, 1))

    # Create a scatter plot
    fig = px.scatter(x=X_range.ravel(), y=y_predicted.ravel(),
                    opacity=0.8, color_discrete_sequence=['black'],
                    labels=dict(x="Value of Humidity3pm", y="Predicted Probability of Rain Tomorrow",))

    # Change chart background color
    fig.update_layout(dict(plot_bgcolor='white'))

    # Update axes lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                    zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                    showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                    zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                    showline=True, linewidth=1, linecolor='black')

    # Set figure title
    fig.update_layout(title=dict(text="Feed Forward Neural Network (1 Input) Model Results",
                                font=dict(color='black')))
    # Update marker size
    fig.update_traces(marker=dict(size=7))

    fig.show()
            
if __name__ == "__main__":
    main()
