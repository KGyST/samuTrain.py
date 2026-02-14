#!/usr/bin/env python3
"""
Test Calamari Learner
"""

import os
import sys
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
import keras
import tf_keras
sys.modules['tensorflow.keras'] = keras
sys.modules['tf_keras'] = tf_keras

# Add src to path
sys.path.insert(0, 'src')

from engines.calamari_learner import CalamariLearner

if __name__ == "__main__":
    learner = CalamariLearner()
    print(f"Model path: {learner.model_path}")
    print(f"Is available: {learner.is_available()}")
    if learner.is_available():
        result = learner.predict('data/single_case/test001.bin.png')
        print(f"Prediction: {result}")
