
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src import tfutility

def  queryDeviceTest():
    print tfutility.get_available_devices()
    print tfutility.get_available_cpus()
    print tfutility.get_available_gpus()

if __name__ == "__main__":
    queryDeviceTest()