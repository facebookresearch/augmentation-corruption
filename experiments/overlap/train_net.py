# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
import os
import time
import datetime

log = logging.getLogger(__name__)

def eta_str(eta_td):
    """Converts an eta timedelta to a fixed-width string format."""
    days = eta_td.days
    hrs, rem = divmod(eta_td.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return '{0:02},{1:02}:{2:02}:{3:02}'.format(days, hrs, mins, secs)

def train_net(model, optimizer, train_dataset,
        batch_size,
        max_epoch,
        loader_params,
        lr_policy,
        checkpoint_folder='checkpoints',
        name=None,
        save_period=1,
        weights=None,
        num_gpus=1,
        is_leader=True):

        chpk_pre = 'model_epoch_'
        if name is not None:
            chpk_pre = name + "_" + chpk_pre
        chpk_post = '.pyth'
        if os.path.exists(checkpoint_folder):
            checkpoints = [c for c in os.listdir(checkpoint_folder) if chpk_post in c and chpk_pre == "_".join(c.split("_")[:-1]) +"_"]
        else:
            checkpoints = []
        if weights:
            checkpoint = torch.load(weights, map_location='cpu')
            log.info("Pretrained weights provided.  Loading model from {} and skipping training.".format(weights))
            if num_gpus > 1:
                model.module.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint['model_state'])

            return model
        elif checkpoints:
            last_checkpoint_name = os.path.join(checkpoint_folder, sorted(checkpoints)[-1])
            checkpoint = torch.load(last_checkpoint_name, map_location='cpu')
            log.info("Loading model from {}".format(last_checkpoint_name))
            if num_gpus > 1:
                model.module.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            start_epoch = 1

        if train_dataset is None:
            return model

        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)\
                if num_gpus > 1 else None
        loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True if sampler is None else False,
                sampler=sampler,
                num_workers=loader_params.num_workers,
                pin_memory=loader_params.pin_memory,
                drop_last=True
                )

        for i in range(start_epoch, max_epoch+1):
            log.info("Starting epoch {}/{}".format(i, max_epoch))
            time_start = time.time()
            if sampler:
                sampler.set_epoch(i)
            train_epoch(model, optimizer, loader, lr_policy, i, num_gpus)
            time_stop = time.time()
            seconds_taken = (time_stop - time_start)
            eta_td = datetime.timedelta(seconds=int(seconds_taken*(max_epoch-i)))
            log.info("Seconds taken: {:.2f}, Time remaining: {}".format(seconds_taken, eta_str(eta_td)))
            
            
            if (i % save_period == 0 or i == max_epoch) and is_leader:
                if num_gpus > 1:
                    m = model.module
                else:
                    m = model

                checkpoint = {
                        'epoch' : i,
                        'model_state' : m.state_dict(),
                        'optimizer_state' : optimizer.state_dict()
                        }
                checkpoint_file = "{:s}{:04d}{:s}".format(chpk_pre, i, chpk_post)
                if not os.path.exists(checkpoint_folder):
                    os.mkdir(checkpoint_folder)
                checkpoint_file = os.path.join(checkpoint_folder, checkpoint_file)
                log.info("Saving model to {}".format(checkpoint_file))
                torch.save(checkpoint, checkpoint_file)

def train_epoch(model, optimizer, loader, lr_policy, epoch, num_gpus=1):
    lr = lr_policy(epoch-1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    model.train()
    loss_fun = torch.nn.CrossEntropyLoss().cuda()
    avg_loss = 0.0
    num_correct = 0
    num_total = 0
    num_batches = 0
    for cur_iter, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = model(inputs)
        loss = loss_fun(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = torch.sum(torch.argmax(preds, dim=1)==labels)
        if num_gpus > 1:
            torch.distributed.all_reduce(correct)
            torch.distributed.all_reduce(loss)
        avg_loss += loss.item()
        num_correct += correct.item()
        num_total += labels.size(0) * num_gpus
        num_batches += num_gpus

    avg_loss /= num_batches
    err = 100 * (1 - num_correct / num_total)
    log.info("Avg loss: {:.3f}, Avg err: {:.3f}".format(avg_loss, err))



    

