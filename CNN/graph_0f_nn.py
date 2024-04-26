
import tensorflow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
model = tensorflow.keras.models.load_model('cnn_werk.keras')
tensorflow.keras.utils.plot_model(model, to_file='cnn_pos.png', show_shapes=True)