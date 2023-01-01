main = "../src/si"
import numpy as np
from si.data.dataset_module import Dataset
from si.neural_networks.layer import Dense, SigmoidActivation, SoftMaxActivation, ReLUActivation, LinearActivation
from si.neural_networks.nn import NN

# Aula 9 - Exercício 10 continuação

# - Exercise 10.3.:Build a neural network model considering the following topology: 32 features, binary problem,
# 3 Dense layers and using SigmoidActivation as activation layer

layer3_1 = Dense(input_size=32, output_size=32)
layer3_2 = Dense(input_size=32, output_size=16)
layer3_3 = Dense(input_size=16, output_size=1)

layer3_1activation = SigmoidActivation()
layer3_2activation = SigmoidActivation()
layer3_3activation = SigmoidActivation()

model10_3 = NN(layers=[layer3_1, layer3_1activation, layer3_2, layer3_2activation, layer3_3, layer3_3activation])

# - Exercise 10.4.:Build a neural network model considering the following topology: 32 features, multiclass problem
# (with 3 classes), 3 Dense layers and using SigmoidActivation as activation layer and SoftMaxActivation as last
# layer of activation

layer4_1 = Dense(input_size=32, output_size=32)
layer4_2 = Dense(input_size=32, output_size=16)
layer4_3 = Dense(input_size=16, output_size=3) # 3 because we have 3 classes

layer4_1activation = SigmoidActivation()
layer4_2activation = SigmoidActivation()
layer4_3activation = SoftMaxActivation()

model10_4 = NN(layers=[layer4_1, layer4_1activation, layer4_2, layer4_2activation, layer4_3, layer4_3activation])

# - Exercise 10.5.:Build a neural network model considering the following topology: 32 features, regression problem,
# 3 Dense layers and using ReLUActivation as activation layer and LinearActivation as last layer of activation

layer5_1 = Dense(input_size=32, output_size=32)
layer5_2 = Dense(input_size=32, output_size=16)
layer5_3 = Dense(input_size=16, output_size=1)

layer5_1activation = ReLUActivation()
layer5_2activation = ReLUActivation()
layer5_3activation = LinearActivation()