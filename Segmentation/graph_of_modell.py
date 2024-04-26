import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

def load_and_plot_model(model_path):
    # Load the Keras model
    model = load_model(model_path)

    # Plot the model architecture
    plot_model(model, to_file='segmentation_arch.png', show_shapes=True, show_layer_names=True)

if __name__ == "__main__":
    model_path = 'segmentation.keras'  # Provide the path to your saved Keras model
    load_and_plot_model(model_path)