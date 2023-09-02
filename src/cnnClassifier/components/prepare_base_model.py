import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = models.vgg16(
            pretrained=self.config.params_weights == 'imagenet',
            num_classes=self.config.params_classes if self.config.params_include_top else 1000
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for idx, layer in enumerate(model.children()):
                if idx < len(model.children()) - freeze_till:
                    for param in layer.parameters():
                        param.requires_grad = False

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, classes)

        full_model = model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(full_model.parameters(), lr=learning_rate)

        return full_model, criterion, optimizer

    def update_base_model(self):
        self.full_model, criterion, optimizer = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path, model):
        torch.save(model.state_dict(), path)