# setup environment variables of MATCH and TVM
import os
CURR_PATH = "./" 
from utils.dump import sanitize_model_to_uint8
# MATCH imports
import match
from pulp_open.pulp_open import PulpOpen
from match.utils.utils import get_default_inputs
from match.model.model import MatchModel

import numpy as np
import json

import torch
import pytorch_benchmarks.image_classification as icl
import matplotlib.pyplot as plt
import netron

from IPython.display import HTML

CHECKPOINTS_DIR = CURR_PATH+"/checkpoints/03"
INPUT_FILE_PATH = CHECKPOINTS_DIR+"/input.txt"
UNSANTIZED_ONNX_FILE_PATH = CHECKPOINTS_DIR+"/GraphModule.onnx"
ONNX_FILE_PATH = CHECKPOINTS_DIR+"/oenne.onnx"
sanitize_model_to_uint8(model_path=UNSANTIZED_ONNX_FILE_PATH, new_model_path=ONNX_FILE_PATH)

datasets = icl.get_data()
dataloaders = icl.build_dataloaders(datasets, batch_size=1)
train_dl, val_dl, test_dl = dataloaders

image, label = next(iter(train_dl))
# Open and load the scaling value
with open(CHECKPOINTS_DIR+'/rescaling_values.json', 'r') as file:
    scaling_values = json.load(file)

# Fill this value with the ones found in the hands-on 3
clip_value = scaling_values['clip_val']
scale_factor = (2**8 - 1) / (clip_value)

def integerize_data(data, clip_value, scale_factor):
    data = torch.clamp(torch.tensor(data), 0, clip_value)
    data = torch.floor(scale_factor * data)
    return data


data_integer = integerize_data(image, clip_value, scale_factor)
np.savetxt(INPUT_FILE_PATH, data_integer.numpy().flatten(), delimiter=',', fmt='%d', header="Input data (3,32,32)")
# Create a figure and set of subplots

# look at the relay network by using the get_relay_network. 
relay_mod, relay_params = match.get_relay_network(filename=ONNX_FILE_PATH)


oenne_model = MatchModel(
    relay_mod = relay_mod,
    relay_params = relay_params,
    model_name = "oenne", executor="graph",
    default_inputs = get_default_inputs(mod=relay_mod, params=relay_params, input_files=[INPUT_FILE_PATH]),
    handle_out_fn="handle_int_classifier",
)
match.match(
    model = oenne_model,
    target = PulpOpen(),
    output_path = CURR_PATH+"/output",
)
