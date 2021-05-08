import tensorflow.keras as keras
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import PARSER


args = PARSER.parse_args()
encoder_path_name = "results/{}/tf_vaegan/tf_vaegan_E_".format(args.env_name)
print('#########',encoder_path_name)

model = tf.keras.models.load_model(encoder_path_name)

for layer in model.layers:
        for weight in layer.weights:
            print(weight.name,weight.shape)