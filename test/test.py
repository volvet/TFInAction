
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src import tfutility

if __name__ == "__main__":
    print tfutility.get_available_devices()
    print tfutility.get_available_cpus()
    print tfutility.get_available_gpus()