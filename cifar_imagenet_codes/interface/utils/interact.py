import os
import logging
from torch.utils.tensorboard import SummaryWriter
import socket


def set_logger(fname):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(fname, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)


class Interact(object):
    def __init__(self, fname_log, summary_root=None, period=None, reported_keys=None):
        r"""
        Args:
            period: the period to report statistics
        """
        self.fname_log = fname_log
        self.summary_root = summary_root
        self.period = period
        self.reported_keys = reported_keys
        os.makedirs(os.path.dirname(self.fname_log), exist_ok=True)
        set_logger(self.fname_log)

        self.writer = None
        if self.summary_root is not None:
            os.makedirs(self.summary_root, exist_ok=True)
            self.writer = SummaryWriter(self.summary_root)

    def report_train(self, statistics, it):
        if it % self.period == 0:
            if self.writer is not None:
                for k, v in statistics.items():
                    self.writer.add_scalar(k, v, global_step=it)
            reported_keys = statistics.keys() if self.reported_keys is None else self.reported_keys
            statistics_str = {k: "{:.5e}".format(float(statistics[k])) for k in reported_keys}
            logging.info("[train] [it: {}] [{}]".format(it, statistics_str))

    def report_val(self, scalar, it):
        self.report_scalar(scalar, it, "val")

    def report_scalar(self, scalar, it, tag):
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar, global_step=it)
        logging.info("[{}] [it: {}] [{:.5e}]".format(tag, it, scalar))

    def report_machine(self):
        logging.info("running @ {}".format(socket.gethostname()))
