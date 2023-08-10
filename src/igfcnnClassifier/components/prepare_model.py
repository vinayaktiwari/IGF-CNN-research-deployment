import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from src.igfcnnClassifier.entity.config_entity import PrepareBaseModelConfig
from keras.layers import Conv2D, Flatten, Dense, Input, Lambda 
from keras import Model
from scipy import fftpack, ifft
import numpy as np
import ast
from keras.models import save_model as keras_save_model
import json

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    @staticmethod
    def dftuv(M,N):
        s = []
        d = []
        a = list(range(0, M))
        c = list(range(0, N))
        idx = PrepareBaseModel.indices(a, lambda x: x > M / 2)
        bv = PrepareBaseModel.indices(a, lambda x: x <= M / 2)
        for i in bv:
            s.append(i)
        for i in idx:
            d.append(i - M)
        u = s + d
        
        b = []
        m = []
        idy = PrepareBaseModel.indices(c, lambda x: x > N / 2)
        bw = PrepareBaseModel.indices(c, lambda x: x <= N / 2)
        for i in bw:
            b.append(i)
        for i in idy:
            m.append(i - N)
        v = b + m
        [V, U] = np.meshgrid(v, u)
        return [V, U]

    @staticmethod
    def lpfilter(M,N,D0):
        [U, V] = PrepareBaseModel.dftuv(M, N)
        D = np.sqrt(U**2 + V**2)
        H = np.exp(-((D**2)/(2*(D0**2))))
        return H

    
    @staticmethod
    def indices(a, func):
        return [i for (i, val) in enumerate(a) if func(val)]
    import tensorflow as tf


    @staticmethod
    def Fourier_decomposition1(img):
        FIBFs = []
        for i in range(1, 6):
            D0 = i * 0.007 * img.shape[1]
            H = PrepareBaseModel.lpfilter(img.shape[0], img.shape[1], D0)

            # Convert the input image to a complex tensor with real part as img and imaginary part as zeros
            img_complex = tf.complex(img, tf.zeros_like(img, dtype=tf.float32))

            # Perform the forward Fourier transform using TensorFlow's FFT
            F = tf.signal.fft2d(img_complex)

            # Perform the element-wise multiplication instead of complex multiplication
            LPF_imageSignal_complex = H * F

            # Take the inverse Fourier transform using TensorFlow's IFFT
            LPF_imageSignal_complex = tf.signal.ifft2d(LPF_imageSignal_complex)

            # Convert the complex tensor back to a real-valued tensor by taking the real part
            LPF_imageSignal = tf.math.real(LPF_imageSignal_complex)

            lpf = LPF_imageSignal[..., 0]

            # Append the LPF_imageSignal to FIBFs without the channel dimension
            FIBFs.append(lpf)

            # Subtract the LPF image signal from the original image
            img = tf.math.real(img) - LPF_imageSignal

        # Append the final img to FIBFs without the channel dimension
        FIBFs.append(tf.math.real(img)[..., 0])
        return FIBFs
    
    @staticmethod
    def Fourier_decomposition(img):
        FIBFs = []
        for i in range(1, 6):
            D0 = i * 0.007 * img.shape[1]
            H = PrepareBaseModel.lpfilter(img.shape[0], img.shape[1], D0)
            img_complex = img
            F = img_complex
            LPF_imageSignal = H * F
            lpf = LPF_imageSignal[..., 0]
            FIBFs.append(lpf)
            img = img - LPF_imageSignal
        FIBFs.append(img[..., 0])
        return FIBFs


    @staticmethod
    def feat_concat(img, size, actual_image_size):
        # Expand the single-channel image to a three-channel image
        img_expanded = tf.image.resize(img, (actual_image_size, actual_image_size))
        img_pseudo_bgr = tf.concat([img_expanded, img_expanded, img_expanded], axis=-1)
        img_gray = tf.image.rgb_to_grayscale(img_pseudo_bgr)

        print("image_input", img_gray.shape)
        FDM = PrepareBaseModel.Fourier_decomposition(img_gray)

        FDM[0] = tf.expand_dims(FDM[0], axis=-1)
        FDM[1] = tf.expand_dims(FDM[1], axis=-1)
        FDM[2] = tf.expand_dims(FDM[2], axis=-1)
        FDM[3] = tf.expand_dims(FDM[3], axis=-1)
        FDM[4] = tf.expand_dims(FDM[4], axis=-1)
        FDM[5] = tf.expand_dims(FDM[5], axis=-1)

        FDM0 = tf.image.resize(FDM[0], (size, size), method=tf.image.ResizeMethod.BILINEAR)
        FDM1 = tf.image.resize(FDM[1], (size, size), method=tf.image.ResizeMethod.BICUBIC)
        FDM2 = tf.image.resize(FDM[2], (size, size), method=tf.image.ResizeMethod.BICUBIC)
        FDM3 = tf.image.resize(FDM[3], (size, size), method=tf.image.ResizeMethod.BICUBIC)
        FDM4 = tf.image.resize(FDM[4], (size, size), method=tf.image.ResizeMethod.BICUBIC)
        FDM5 = tf.image.resize(FDM[5], (size, size), method=tf.image.ResizeMethod.BICUBIC)

        # Convert tensors to real-valued tensors using tf.math.real
        FDM_components = tf.stack(
            [
                tf.math.real(FDM1),
                tf.math.real(FDM2),
                tf.math.real(FDM3),
                tf.math.real(FDM4),
                tf.math.real(FDM5),
            ],
            axis=0,
        )

        FDM_components = tf.transpose(FDM_components, perm=[3, 1, 2, 0])
        FDM_components.set_shape([None, size, size, 5])

        return FDM_components


    @staticmethod
    def IgfCNN(input_shape:tuple, num_classes:int, size:int, learning_rate:float):
        input_shape = tuple(input_shape)
        inputs = Input(shape=input_shape)    
        img_reshaped = Lambda(lambda x: tf.reshape(x, (-1,) +input_shape[1:]))(inputs)
        print("================image_reshaped=============", img_reshaped)
        processed_images = PrepareBaseModel.feat_concat(img_reshaped, size,input_shape[0])

        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(processed_images)
    #     x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(inputs)
    #     x = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(x)
        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    #     x = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(x)
        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    #     x = MaxPooling2D(pool_size=(4, 4), strides=(1, 1))(x)
        
        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    #     x = MaxPooling2D(pool_size=(4, 4), strides=(1, 1))(x)
        
        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    #     x = MaxPooling2D(pool_size=(4, 4), strides=(1, 1))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(16, activation='relu')(x)

        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

        model.summary()
        return model


    def update_base_model(self):
        image_size_tuple = ast.literal_eval(self.config.params_image_size)

        print("Type of params_image_size:", type(image_size_tuple))
        print("params_image_size:", image_size_tuple)


        self.model = self.IgfCNN(
            input_shape = image_size_tuple,
            num_classes = self.config.params_classes, 
            size=50,
            learning_rate=self.config.params_learning_rate)
        
        return self.model
        # self.save_model(path=self.config.updated_base_model_path, model=self.model)

    # @staticmethod
    # def save_model(path: Path, model: tf.keras.Model):
    #     model.save(path)
