#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tfaip').setLevel(logging.ERROR)
logging.getLogger('calamari_ocr').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.scripts.train import main

if __name__ == '__main__':
    # Create default trainer params
    trainer_params = CalamariScenario.default_trainer_params()

    # Set the training parameters equivalent to the command line
    trainer_params.output_dir = "models/new_model"
    trainer_params.epochs = 5
    trainer_params.gen.train.images = ["data/64_case/*.bin.png", ]
    trainer_params.gen.val.images = ["data/single_case/*.bin.png"]
    trainer_params.gen.setup.train.batch_size = 1
    trainer_params.gen.setup.train.num_processes = 1
    trainer_params.gen.setup.val.num_processes = 1
    
    # Performance optimizations for Windows
    trainer_params.progress_bar = True

    # Run the training
    result = main(trainer_params)
    print("Training completed.")
