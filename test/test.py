
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src import tfutility

import linear_regression_test as lr

def  queryDeviceTest():
    tfutility.get_available_devices()
    tfutility.get_available_cpus()
    tfutility.get_available_gpus()
    print "queryDeviceTest:    [OK]"

if __name__ == "__main__":
    queryDeviceTest()
    lr.linear_regression_test()