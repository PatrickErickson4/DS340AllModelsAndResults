import os
from Archive.dataset import load_dataset
from Archive.utils import print_config
from model import build_model
import tensorflow as tf
from tensorflow import keras
import json
import tensorflow_hub as hub



train_generator, valid_generator, test_generator = load_dataset()
print("\n\n______________CLASS INDICES TO NAME MAPPING_______________")
print_config(train_generator.class_indices)
print("___________________________________________________________\n\n")



config = json.load(open('config.json', 'r'))
    # Loading the Saved Model
try:
    pretrained_model = keras.models.load_model(os.path.join(config['checkpoint_filepath'], 'model.h5'))
except ValueError as e:
    if 'Unknown layer: KerasLayer' in str(e):
        pretrained_model = keras.models.load_model(os.path.join(config['checkpoint_filepath'], 'model.h5'),
                        custom_objects={'KerasLayer': hub.KerasLayer})
    else:
        raise
print(pretrained_model.summary())

# predictions = tf.argmax(pretrained_model.predict(test_generator), axis=1)

result = pretrained_model.evaluate(test_generator)
print(result)

