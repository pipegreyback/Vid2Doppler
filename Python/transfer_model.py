import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from pathlib import Path

parser.add_argument("--epochs", type=int, default="10", help="Number of epochs. Default is 10. ")
parser.add_argument("--model_directory", type=str, help="Route to the already trained model. ")
parser.add_argument("--training_directory", type=str, default=training_directory, help="The dir for training data. ")
parser.add_argument("--validation_directory", type=str, default=validation_directory, help="The dir for validation data. ")
parser.add_argument("--testing_directory", type=str, default=testing_directory, help="The dir for testing data. ")
parser.add_argument("--saved_model_directory", type=str, default=saved_model_dir, help="The dir for the saved model. ")


class Trainer():
  def simple_transfer(self):
    optim_2 = Adam(lr=0.0001)

    # Re-compile the model, this time leaving the last 2 layers unfrozen for Fine-Tuning
    vgg_model_ft = create_model(input_shape, n_classes, optim_2, fine_tune=2)

    vgg_ft_history = vgg_model_ft.fit(traingen,
                                      epochs=epochs,
                                      validation_data=validgen,
                                      steps_per_epoch=n_steps, 
                                      validation_steps=n_val_steps,
                                      callbacks=[tl_checkpoint_1, early_stop, plot_loss_2],
                                      verbose=1)

    vgg_model_ft.load_weights('tl_model_v1.weights.best.hdf5') # initialize the best trained weights

    vgg_preds_ft = vgg_model_ft.predict(testgen)
    vgg_pred_classes_ft = np.argmax(vgg_preds_ft, axis=1)
    vgg_acc_ft = accuracy_score(true_classes, vgg_pred_classes_ft)
  
  def transfer_model(self):
    model = create_model(self.__input_window_length)
    model.load_weights(self.__saved_model_dir)

    # Freeze convolutional layers
    for layer in model.layers[:-2]:
      layer.trainable = False

     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.__learning_rate, beta_1=self.__beta_1, beta_2=self.__beta_2), loss=self.__loss, metrics=self.__metrics) 
     early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=self.__min_delta, patience=self.__patience, verbose=self.__verbose, mode="auto")
