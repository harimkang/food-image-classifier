from unittest import TestCase

import os

from tensorflow.python.eager.context import num_gpus

from model import Inception_v3

class ModelTest(TestCase):
    
    def test_model_generate(self):
        
        img_width, img_height = 299, 299
        class_list = 'tests/assets/classes.txt'
        with open(class_list, 'r') as txt:
            label = [read.strip() for read in txt.readlines()]
        
        model = Inception_v3(class_list=label, img_width=img_width, img_height=img_height,
                             batch_size=16)
        
        self.assertIsNotNone(model)