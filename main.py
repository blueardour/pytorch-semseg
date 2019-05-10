import os, sys, glob, time, yaml
import numpy as np
import logging
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
import utils
from tensorboardX import SummaryWriter

def main():
    args = utils.get_config()

    if utils.check_file(args.config):
        with open(args.config) as fp:
            if hasattr(yaml, 'FullLoader'):
                cfg = yaml.load(fp, Loader=yaml.FullLoader)
            else:
                cfg = yaml.load(fp)

    # log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if isinstance(args.model, str):
        model_arch = args.model
        model_name = model_arch
    else:
        model_name = args.model['arch']
        model_arch = args.model['base'] + '-' + args.model['arch']

    if args.evaluate:
        log_suffix = model_arch + '-eval-' + args.case
    else:
        log_suffix = model_arch + '-' + args.case
    utils.setup_logging(os.path.join(args.log_dir, log_suffix + '.txt'), resume=args.resume)

    # tensorboard
    if args.tensorboard and not args.evaluate:
        args.tensorboard = SummaryWriter(args.log_dir, filename_suffix='.' + log_suffix)
    else:
        args.tensorboard = None

    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and len(args.device_ids) > 0:
        args.device_ids = [x for x in args.device_ids if x < torch.cuda.device_count() and x >= 0]
        if len(args.device_ids) == 0:
            args.device_ids = None
    else:
        args.device_ids = None

    if args.device_ids is not None:
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic=True #https://github.com/pytorch/pytorch/issues/8019

    logging.info("device_ids: %s" % args.device_ids)
    logging.info("no_decay_small: %r" % args.no_decay_small)
    logging.info("optimizer: %r" % args.optimizer)
    #logging.info(type(args.lr_custom_step))
    #if type(args.lr_custom_step) is str:
    args.lr_custom_step = [int(x) for x in args.lr_custom_step.split(',')]
    logging.info("lr_custom_step: %r" % args.lr_custom_step)

    if model_name in models.model_zoo:
        model = models.get_model(args, cfg['n_classes'])
    else:
        logging.error("model(%s) not support, available models: %r" % (model_name, models.model_zoo))
        return
    criterion = models.get_loss_function(cfg)

    utils.check_folder(args.weights_dir)
    args.weights_dir = os.path.join(args.weights_dir, model_name)
    utils.check_folder(args.weights_dir)
    args.resume_file = os.path.join(args.weights_dir, args.case + "-" + args.resume_file)
    args.pretrained = os.path.join(args.weights_dir, args.pretrained)
    epoch = 0
    best_acc = 0
    # resume training
    if args.resume:
        logging.info("resuming from %s" % args.resume_file)
        checkpoint = torch.load(args.resume_file)
        epoch = checkpoint['epoch']
        logging.info("resuming ==> last epoch: %d" % epoch)
        epoch = epoch + 1
        best_acc = checkpoint['best_acc']
        logging.info("resuming ==> best_acc: %f" % best_acc)
        utils.load_state_dict(model, checkpoint['state_dict'])
        logging.info("resumed from %s" % args.resume_file)
    else:
        if utils.check_file(args.pretrained):
            logging.info("resuming from %s" % args.pretrained)
            checkpoint = torch.load(args.pretrained)
            logging.info("resuming ==> last epoch: %d" % checkpoint['epoch'])
            logging.info("resuming ==> last best_acc: %f" % checkpoint['best_acc'])
            logging.info("resuming ==> last learning_rate: %f" % checkpoint['learning_rate'])
            utils.load_state_dict(model, checkpoint['state_dict'])
        else:
            logging.info("no pretrained file exists({}), init model with default initlizer".
                format(args.pretrained))

    if args.device_ids is not None:
        torch.cuda.set_device(args.device_ids[0])
        if not isinstance(model, nn.DataParallel) and len(args.device_ids) > 1:
            model = nn.DataParallel(model, args.device_ids).cuda()
        else:
            model = model.cuda()
        #if criterion is not None:
        #    criterion = criterion.cuda()

    # dataset
    data_path = cfg["data"]["path"]
    dataset = cfg["data"]["dataset"]
    logging.info("loading dataset with batch_size {} and val-batch-size {}. dataset {} path: {}".
        format(args.batch_size, args.val_batch_size, dataset, data_path))

    data_loader = models.data_loader(dataset)

    if args.val_batch_size < 1:
        val_loader = None
    else:
        val_dataset = data_loader(
            data_path, split=cfg['data']["val_split"], sbd_path=cfg["data"]["sbd_path"],
            is_transform=True, img_size=(cfg['data']["img_rows"], cfg['data']["img_cols"]), )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=args.val_batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate and val_loader is not None:
        logging.info("evaluate the dataset on pretrained model...")
        score, class_iou = validate(val_loader, model, criterion, args)
        for k, v in score.items():
            logger.info("{}: {}".format(k, v))
        for k, v in class_iou.items():
            logger.info("{}: {}".format(k, v))
        return

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = models.get_composed_augmentations(augmentations)
    #if data_aug is None:
    #    logging.info("No data augmentations, consirder one? Here are some available: %r" %
    #        list(models.key2aug.keys()))

    train_dataset = data_loader(
        data_path,  split=cfg["data"]["train_split"], sbd_path=cfg["data"]["sbd_path"],
        is_transform=True,
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
    )

    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        shape = value.shape
        if args.no_decay_small and ((len(shape) == 4 and shape[1] == 1) or (len(shape) == 1)):
            params += [{'params':value, 'weight_decay':0}]
        else:
            params += [{'params':value}]

    optimizer = None
    if args.optimizer == "ADAM":
        optimizer = torch.optim.Adam(params, lr=args.lr)

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    logging.info("start to train network " + model_name + ' with case ' + args.case)
    while epoch < args.epochs:
        lr = utils.adjust_learning_rate(optimizer, epoch, args)
        logging.info('[epoch %d]: lr %e', epoch, lr)

        # training
        loss = train(train_loader, model, criterion, optimizer, args)
        logging.info('[epoch %d]: train_loss %.3f' % (epoch, loss))

        # validate
        score, class_iou = validate(val_loader, model, criterion, args)
        val_acc = score["Mean IoU : \t"]
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        logging.info('[epoch %d]: current acc: %f, best acc: %f', epoch, val_acc, best_acc)

        if args.tensorboard is not None:
            args.tensorboard.add_scalar(log_suffix + '/eval-acc', val_acc, epoch)
            args.tensorboard.add_scalar(log_suffix + '/train-loss', loss, epoch)
            args.tensorboard.add_scalar(log_suffix + '/lr', lr, epoch)

        utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_acc': best_acc,
                'learning_rate': lr,
                }, is_best, args)

        epoch = epoch + 1


def train(loader, model, criterion, optimizer, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(loader):
        data_time.update(time.time() - end)
        if args.device_ids is not None:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        outputs = model(input)

        loss = criterion(outputs, target)

        if i % args.iter_size == 0:
            optimizer.zero_grad()

        loss.backward()

        if i % args.iter_size == (args.iter_size - 1):
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) 
            optimizer.step()

        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.report_freq == 0:
            logging.info('train %d/%d, loss:%.3f(%.3f), batch time:%.1f(%.1f), data load time: %.1f(%.1f)' %
              (i, len(loader), losses.val, losses.avg,
               batch_time.val,  batch_time.avg, data_time.val, data_time.avg))

    return losses.avg

def validate(loader, model, criterion, args):
    if loader is None:
        logging.info('eval_loader is None, skip validate')
        return None, None

    running_metrics_val = models.runningScore(args.n_classes)
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(loader):

            if args.device_ids is not None:
                input = input.cuda(non_blocking=True)
                #target = target.cuda(non_blocking=True)
          
            outputs = model(input)

            if args.eval_flip and not args.evaluate:
                # Flip images in numpy (not support in tensor)
                outputs = outputs.data.cpu().numpy()
                flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
                flipped_images = torch.from_numpy(flipped_images).float().cuda(non_blocking=True)
                outputs_flipped = model(flipped_images)
                outputs_flipped = outputs_flipped.data.cpu().numpy()
                outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0
                pred = np.argmax(outputs, axis=1)
            else:
                pred = outputs.data.max(1)[1].cpu().numpy()

            gt = target.data.cpu().numpy()
            running_metrics.update(gt, pred)

        score, class_iou = running_metrics_val.get_scores()

    return score, class_iou
 

if __name__ == '__main__':
    main()

