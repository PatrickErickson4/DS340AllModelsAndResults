import os
import json
import pandas as pd
from Archive.utils import print_config # comment this out to run main
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt

def applyCLAHE(image, display: bool = False):
    """
    CLAHE implementation that is agnostic to whether `image` is in [0,1] or [0,255].
    Always returns a float32 image in [0,1].
    """

    img = image.astype(np.float32)

    if img.max() <= 1.0:
        img *= 255.0

    img_u8 = np.clip(img, 0, 255).astype(np.uint8)

    # 4) RGB → LAB
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(7,7))
    cl = clahe.apply(l)

    # 6) Merge & LAB → RGB
    merged    = cv2.merge([cl, a, b])
    rgb_u8    = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    # 7) Back to float32 [0,1]
    out = rgb_u8.astype(np.float32) / 255.0

    # 8) (Optional) Display before/after
    if display:
        fig, axes = plt.subplots(1, 2, figsize=(8,4), dpi=200)
        axes[0].imshow(img_u8);    axes[0].set_title("input uint8"); axes[0].axis("off")
        axes[1].imshow(out);       axes[1].set_title("CLAHE float"); axes[1].axis("off")
        plt.show()

    return out

def load_dataset(config_file="config.json"):
    config = json.load(open(config_file, "r"))
    data_aug = config["data_augmentations"]

    if data_aug["isVIT"]:
        target_size = (256, 256)
    else:
        target_size = (config["img_height"], config["img_width"])

    preprocessing_fn = applyCLAHE if data_aug["applyCLAHE"] else None

    # changed: choose whether to rescale based on if CLAHE is applied
    rescale_factor = None if data_aug["applyCLAHE"] else 1.0 / 255  # changed

    aug_data_generator = ImageDataGenerator(
        rotation_range=data_aug['rotation_range'],
        horizontal_flip=data_aug['horizontal_flip'],
        width_shift_range=data_aug['width_shift_range'],
        height_shift_range=data_aug['height_shift_range'],
        shear_range=data_aug['shear_range'],
        preprocessing_function=preprocessing_fn,
        rescale=rescale_factor,  # changed
    )

    reg_data_generator = ImageDataGenerator(
        preprocessing_function=preprocessing_fn,
        rescale=rescale_factor,  # changed
    )

    if data_aug["TRAIN_AUG"]:
        data_generator = aug_data_generator
        print('[INFO] Augmentation is applied on training data generator')
    else:
        data_generator = reg_data_generator
        print('[INFO] No Augmentation is applied on training data generator')
    train_generator = data_generator.flow_from_dataframe(
        pd.read_csv(os.path.join(config["dataset_dir"], "train.csv")),
        directory=None,
        x_col='filepath',
        y_col='label_tag',
        target_size=target_size,
        batch_size=config["batch_size"],
        shuffle=True,
        class_mode='sparse'
    )

    if data_aug["VALID_AUG"]:
        data_generator = aug_data_generator
        print('[INFO] Augmentation is applied on validation data generator')
    else:
        data_generator = reg_data_generator
        print('[INFO] No Augmentation is applied on validation data generator')
    valid_generator = data_generator.flow_from_dataframe(
        pd.read_csv(os.path.join(config["dataset_dir"], "valid.csv")),
        directory=None,
        x_col='filepath',
        y_col='label_tag',
        target_size=target_size,
        batch_size=config["batch_size"],
        shuffle=True,
        class_mode='sparse'
    )

    if data_aug["TEST_AUG"]:
        data_generator = aug_data_generator
        print('[INFO] Augmentation is applied on Test data generator')
    else:
        data_generator = reg_data_generator
        print('[INFO] No Augmentation is applied on Test data generator')
    test_generator = data_generator.flow_from_dataframe(
        pd.read_csv(os.path.join(config["dataset_dir"], "test.csv")),
        directory=None,
        x_col='filepath',
        y_col='label_tag',
        target_size=target_size,
        batch_size=config["batch_size"],
        shuffle=False,
        class_mode='sparse'
    )

    return train_generator, valid_generator, test_generator


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(100)
    # 1. Load the generators
    train_gen, valid_gen, test_gen = load_dataset("VITCLAHE.json")

    # 2. Grab one batch of images + labels
    x_batch, y_batch = next(train_gen)   # x_batch.shape == (batch_size, H, W, 3)

    # 3. Plot the first N images
    N = 4
    fig, axes = plt.subplots(1, N, figsize=(4*N, 4))
    for i in range(N):
        axes[i].imshow(x_batch[i])
        axes[i].set_title(f"label={y_batch[i]}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()