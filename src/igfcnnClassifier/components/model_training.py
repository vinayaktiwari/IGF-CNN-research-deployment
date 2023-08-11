import numpy as np
from src.igfcnnClassifier.entity.config_entity import TrainingConfig
from src.igfcnnClassifier.components.prepare_model import PrepareBaseModel
import ast
import tensorflow as tf
from pathlib import Path
from src.igfcnnClassifier.entity.config_entity import PrepareBaseModelConfig
from keras.callbacks import EarlyStopping,ReduceLROnPlateau


class Training:
    def __init__(self, config: TrainingConfig, model: PrepareBaseModel, config1: PrepareBaseModelConfig):
        self.config = config
        self.config1 = config1
        self.model = model
    
    # def get_base_model(self):
    #     self.model = tf.keras.models.load_model(
    #         self.config.updated_base_model_path
    #     )
    
    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.10
        )
        dataflow_kwargs = dict(
            # target_size=self.config.params_image_size[:-1],
            target_size=(250,250),
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        image_size_tuple = ast.literal_eval(self.config1.params_image_size)

        learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience =2, verbose=1,factor=0.05)


        model = self.model.IgfCNN(input_shape = image_size_tuple,
            num_classes = self.config1.params_classes, 
            size=50,
            learning_rate=self.config1.params_learning_rate
        )

        model.fit(self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=[learning_rate_reduction])

        self.save_model(
            path=self.config.trained_model_path,
            model=model
        )
