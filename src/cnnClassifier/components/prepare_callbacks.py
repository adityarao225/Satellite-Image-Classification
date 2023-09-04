import os
import urllib.request as request
from zipfile import ZipFile
import torch
import time

import os
import time
from tensorboardX import SummaryWriter
import torch
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
        self.tb_writer = None

    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return SummaryWriter(logdir=tb_running_log_dir)

    def _create_ckpt_callbacks(self):
        return torch.save(self.config.model.state_dict(), self.config.checkpoint_model_filepath)

    def get_tb_ckpt_callbacks(self):
        self.tb_writer = self._create_tb_callbacks()
        return [
            self._create_ckpt_callbacks
        ]