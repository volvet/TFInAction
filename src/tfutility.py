
from tensorflow.python.client import device_lib


def get_available_devices():
    devices = device_lib.list_local_devices()
    return [x.name for x in devices]

def get_available_cpus():
    devices = device_lib.list_local_devices()
    return [x.name for x in devices if x.device_type == 'CPU']

def get_available_gpus():
    devices = device_lib.list_local_devices()
    return [x.name for x in devices if x.device_type == "GPU"]


if __name__ == "__main__":
    print get_available_devices()
    print get_available_cpus()
    print get_available_gpus()