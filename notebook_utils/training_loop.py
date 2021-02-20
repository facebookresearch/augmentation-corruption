# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def train_model(model, dataset, num_workers, gpu_device):
    
    max_epochs = 100
    batch_size = 128
    base_lr = 0.1

    # Cosine learning rate decay
    def get_lr(cur_epoch):
        return 0.5 * base_lr * (1.0 + np.cos(np.pi * cur_epoch / max_epochs))
    
    optim = torch.optim.SGD(model.parameters(),
        lr=base_lr,
        nesterov=True,
        momentum=0.9,
        weight_decay=0.0005,
        dampening=0.0,    
        )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
        )
    
    loss_fun = torch.nn.CrossEntropyLoss().cuda(device=gpu_device)
    
    model.train()
    epoch_loss = 0
    for cur_epoch in range(max_epochs):
        #Set learning rate for current epoch
        for param_group in optim.param_groups:
            param_group['lr'] = get_lr(cur_epoch)
            
        for inputs, labels in dataloader:
            inputs = inputs.cuda(device=gpu_device)
            labels = labels.cuda(device=gpu_device, non_blocking=True)
            
            preds = model(inputs)
            loss = loss_fun(preds, labels)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            epoch_loss += loss.item()
            
        epoch_loss /= len(dataloader)
        print("Completed epoch {}.  Average training loss: {}".format(cur_epoch+1, epoch_loss))
    model.eval()
