from multiprocessing import Process
from typing import List
import time
from typing import Union, Tuple
import os


def get_gpu_memory_map():
    raw = list(os.popen('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader'))
    mem = [int(x.strip()) for x in raw]
    return dict(zip(range(len(mem)), mem))


def get_gpu_total_memory_map():
    raw = list(os.popen('nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader'))
    mem = [int(x.strip()) for x in raw]
    return dict(zip(range(len(mem)), mem))


def gpu_memory_consumption():
    devices = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(',')))
    gpu_memory_map = get_gpu_memory_map()
    gpu_total_memory_map = get_gpu_total_memory_map()
    return sum([gpu_total_memory_map[device] - gpu_memory_map[device] for device in devices])


def available_devices(threshold=10000) -> List[int]:
    gpu_memory_map = get_gpu_memory_map()
    devices = []
    for idx, mem in gpu_memory_map.items():
        if mem > threshold:
            devices.append(idx)
    return devices


def format_devices(devices: Union[int, List[int], Tuple[int]]):
    if isinstance(devices, int):
        return "{}".format(devices)
    elif isinstance(devices, tuple) or isinstance(devices, list):
        return ','.join(map(str, devices))


class Task(object):
    def __init__(self, process: Process, n_devices: int = 1):
        self.process = process
        self.n_devices = n_devices
        self.devices = None
        self.just_created = True

    def state(self):
        if self.just_created:
            return 'just_created'
        elif self.process.is_alive():
            return 'is_alive'
        else:
            return 'finished'

    def start(self, devices):
        self.devices = devices
        os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(devices)
        self.process.start()
        self.just_created = False


class DevicesPool(object):
    def __init__(self, devices: List[int]):
        self.devices = devices.copy()

    def flow_out(self, n_devices: int):
        if len(self.devices) < n_devices:
            return None
        ret = []
        for _ in range(n_devices):
            ret.append(self.devices.pop())
        return ret

    def flow_in(self, devices: List[int]):
        for device in devices:
            self.devices.append(device)


################################################################################
# Run multiple tasks run in parallel, exclusively using devices
# Suitable for running tasks consuming high gpu memory
################################################################################

def wait_schedule(tasks: List[Task], devices: List[int]):
    # assert len(set(devices)) == len(devices)
    for task in tasks:
        assert task.n_devices <= len(devices)
    tasks = sorted(tasks, key=lambda x: x.n_devices, reverse=True)
    devices_pool = DevicesPool(devices)

    def linked_list_next(_idx: int, _lst: List):
        if _lst:
            return (_idx + 1) % len(_lst)
        else:
            return -1

    idx = 0
    while tasks:
        task = tasks[idx]
        state = task.state()
        if state == 'just_created':
            devices = devices_pool.flow_out(task.n_devices)
            if devices is not None:
                print("\033[1m start a task with {} devices".format(len(devices)))
                task.start(devices)
        elif state == 'finished':
            print("\033[1m a task with {} devices finished".format(len(task.devices)))
            devices_pool.flow_in(task.devices)
            task.process.close()
            tasks.pop(idx)
            idx -= 1
        idx = linked_list_next(idx, tasks)
        time.sleep(1)
