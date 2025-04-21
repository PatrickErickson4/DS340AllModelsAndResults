import os
# only show errors (0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from absl import logging as absl_logging
# only show errors from Abseil
absl_logging.set_verbosity(absl_logging.ERROR)

import logging
import tensorflow as tf
# only show errors from the tensorflow logger
tf.get_logger().setLevel(logging.ERROR)

import json
import time
import numpy as np
import argparse

from Archive.utils import print_config, load_callbacks, save_training_history, plot_training_summary
from Archive.dataset import load_dataset
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping

def runTrain(config_path):
    # Load configuration
    with open(config_path, 'r') as cfg_file:
        config = json.load(cfg_file)
    print_config(config)

    # Load dataloaders
    train_generator, valid_generator, test_generator = load_dataset(config_path)

    # Build model
    model = build_model(config_path)

    # EarlyStopping: monitor val_accuracy, min_delta=1e-4, patience=10 as done in the parent paper
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        min_delta=1e-4,
        patience=10,
        verbose=1,
        mode='max',
        restore_best_weights=True
    )

    # Combine EarlyStopping with any existing callbacks
    callbacks = load_callbacks(config) + [early_stop]

    # Train model: up to 1000 epochs with early stopping
    start = time.time()
    train_history = model.fit(
        train_generator,
        epochs=config['epochs'],
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        callbacks=callbacks
    )
    end = time.time()

    # Save model outputs for this config
    ckpt_dir = config.get('checkpoint_filepath', 'checkpoints')
    if not os.path.exists(ckpt_dir):
        print(f"[INFO] Creating directory {ckpt_dir} to save the trained model")
        os.makedirs(ckpt_dir)
    print(f"[INFO] Saving the model and log in \"{ckpt_dir}\" directory")
    model.save(os.path.join(ckpt_dir, 'saved_model'))

    # Also save HDF5 copy
    h5_path = os.path.join(ckpt_dir, 'model.h5')
    model.save(h5_path)
    print(f"[INFO] HDF5 model saved to {h5_path}")

    # Save training history
    save_training_history(train_history, config)

    # Plot training history
    plot_training_summary(config)

    # Print training summary
    training_time = end - start
    print(f"[INFO] Total time elapsed for {config_path}: {training_time} seconds")
    print(f"[INFO] Time per epoch: {training_time // config['epochs']} seconds")
