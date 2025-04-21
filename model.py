import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import (
    MobileNetV2,
    DenseNet201,
    ResNet152V2,
    VGG19,
    InceptionV3,
    MobileNetV3Small,
    MobileNetV3Large
)
import tensorflow_hub as hub  # ← added

def build_model(config_file="config.json"):
    config = json.load(open(config_file, "r"))

    # Model Selection
    backbone = None
    backbone_name = config["model_configuration"]["backbone_name"].lower()

    if backbone_name == "mobilenetv2":
        print(f"[INFO]: Selected Model: {backbone_name}")
        backbone = MobileNetV2(
            input_shape=(config["img_width"], config["img_height"], 3),
            include_top=False,
            pooling="max",
            weights="imagenet"
        )

    elif backbone_name == "densenet201":
        print(f"[INFO]: Selected Model: {backbone_name}")
        backbone = DenseNet201(
            input_shape=(config["img_width"], config["img_height"], 3),
            include_top=False,
            pooling="max",
            weights="imagenet"
        )

    elif backbone_name == "resnet152v2":
        print(f"[INFO]: Selected Model: {backbone_name}")
        backbone = ResNet152V2(
            input_shape=(config["img_width"], config["img_height"], 3),
            include_top=False,
            pooling="max",
            weights="imagenet"
        )

    elif backbone_name == "vgg19":
        print(f"[INFO]: Selected Model: {backbone_name}")
        backbone = VGG19(
            input_shape=(config["img_width"], config["img_height"], 3),
            include_top=False,
            pooling="max",
            weights="imagenet"
        )

    elif backbone_name == "inceptionv3":
        print(f"[INFO]: Selected Model: {backbone_name}")
        backbone = InceptionV3(
            input_shape=(config["img_width"], config["img_height"], 3),
            include_top=False,
            pooling="max",
            weights="imagenet"
        )

    elif backbone_name == "mobilenetv3small":
        print(f"[INFO]: Selected Model: {backbone_name}")
        backbone = MobileNetV3Small(
            input_shape=(config["img_width"], config["img_height"], 3),
            include_top=False,
            pooling="max",
            weights="imagenet"
        )

    elif backbone_name == "mobilenetv3large":
        print(f"[INFO]: Selected Model: {backbone_name}")
        backbone = MobileNetV3Large(
            input_shape=(config["img_width"], config["img_height"], 3),
            include_top=False,
            pooling="max",
            weights="imagenet"
        )

    elif backbone_name == "mobilevitsmall":
        print(f"[INFO]: Selected Model: {backbone_name}")
        # Use MobileViT-Small feature extractor from TensorFlow Hub
        hub_url = "https://tfhub.dev/sayannath/mobilevit_s_1k_256_fe/1"
        backbone = hub.KerasLayer(
            hub_url,
            # trained on 256 image size
            input_shape=(256, 256, 3),
            trainable=False,
            name="MobileViT_Small_FeatureExtractor"
        )

    else:
        identifier = config["model_configuration"]["backbone_name"]
        raise ValueError(f"[ERROR]: No application module found with identifier: {identifier}")

    # Setting the transfer learning mode
    backbone.trainable = True

    # Creating Sequential Model
    model = Sequential()
    model.add(backbone)
    if config["add_dense"]:
        model.add(BatchNormalization())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(BatchNormalization())
        model.add(Flatten())
    else:
        model.add(BatchNormalization())
        model.add(Flatten())
    model.add(Dense(config["n_classes"], activation='softmax'))

    # Optimizer selection
    if config["model_configuration"]["optimizer"] == "adam":
        print(f'[INFO]: Selecting Adam as the optimizer')
        print(f'[INFO]: Learning Rate: {config["learning_rates"]["initial_lr"]}')
        opt = Adam(learning_rate=config["learning_rates"]["initial_lr"])
    else:
        opt = SGD()

    # Building the Model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['acc', 'mse']
    )
    return model


if __name__ == "__main__":
    model = build_model()
    print(model)
    model.summary()
