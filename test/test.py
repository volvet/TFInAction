
import sys
import os
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src import tfutility

import linear_regression_test as lr

def  queryDeviceTest():
    print "[Run] queryDeviceTest"
    tfutility.get_available_devices()
    tfutility.get_available_cpus()
    tfutility.get_available_gpus()
    print " [OK] queryDeviceTest"

if __name__ == "__main__":
    queryDeviceTest()
    lr.linear_regression_test()
    lr.linear_regression2_test()

