import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BinaryClassifier(tf.keras.Model):
    """description of class"""

    def __init__(self, model_shape=[1]):
        super(BinaryClassifier, self).__init__()

        self.dense_layers = []
        for units in model_shape:
            self.dense_layers.append(tf.keras.layers.Dense(units, "relu"))

    def call(self, inputs):
        
        output = inputs

        for layer in self.dense_layers:
            output = layer(output)

        return output






