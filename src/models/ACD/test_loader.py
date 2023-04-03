from __future__ import division
from __future__ import print_function

from collections import defaultdict

import time
import numpy as np
import torch

from modules import *
import arg_parser
import logger
import data_loader
import forward_pass_and_eval
import utils_x as utils
import model_loader

args = arg_parser.parse_args()
logs = logger.Logger(args)

if args.GPU_to_use is not None:
    logs.write_to_log_file("Using GPU #" + str(args.GPU_to_use))

(
        train_loader,
        valid_loader,
        test_loader,
        loc_max,
        loc_min,
        vel_max,
        vel_min,
    ) = data_loader.load_data(args)

for i in range(2):
    for batch_idx, minibatch in enumerate(train_loader):
        data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

        print("data: {}".format(data.size()))
        print("relations: {}".format(relations.size()))