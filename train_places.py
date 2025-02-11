import os
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.reactnet_imagenet_ride as reactnet
from parse_config import ConfigParser
from trainer import Trainer
from dataset import get_dataset

deterministic = False
if deterministic:
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

def learing_rate_scheduler(optimizer, config):
    if "type" in config._config["lr_scheduler"]: 
        if config["lr_scheduler"]["type"] == "CustomLR": # linear learning rate decay
            lr_scheduler_args = config["lr_scheduler"]["args"]
            gamma = lr_scheduler_args["gamma"] if "gamma" in lr_scheduler_args else 0.1
            print("Scheduler step1, step2, warmup_epoch, gamma:", (lr_scheduler_args["step1"], lr_scheduler_args["step2"], lr_scheduler_args["warmup_epoch"], gamma))
            def lr_lambda(epoch):
                if epoch >= lr_scheduler_args["step2"]:
                    lr = gamma * gamma
                elif epoch >= lr_scheduler_args["step1"]:
                    lr = gamma
                else:
                    lr = 1

                """Warmup"""
                warmup_epoch = lr_scheduler_args["warmup_epoch"]
                if epoch < warmup_epoch:
                    lr = lr * float(1 + epoch) / warmup_epoch
                return lr
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)  # cosine learning rate decay
    else:
        lr_scheduler = None
    return lr_scheduler

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    cls_num_list = data_loader.cls_num_list
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    # model = config.init_obj('arch', module_arch)
    model = reactnet.reactnet(**dict(config['arch']['args']))
    logger.info(model)
    if os.path.isfile(config['arch']['load_weight']):
        checkpoint = torch.load(config['arch']['load_weight'])
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print('weight load complete')


    # get function handles of loss and metrics
    loss_class = getattr(module_loss, config["loss"]["type"])
    if hasattr(loss_class, "require_num_experts") and loss_class.require_num_experts:
        criterion = config.init_obj('loss', module_loss, cls_num_list=data_loader.cls_num_list, num_experts=config["arch"]["args"]["num_experts"])
    else:
        criterion = config.init_obj('loss', module_loss, cls_num_list=data_loader.cls_num_list)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler.  
    expert_params = []
    linear_params =[]
    for pname, p in model.named_parameters():
        # if  'layer4s'  in pname:
        #     conv4_params += [p]
        # elif 'linears' in pname:
        #     linear_params += [p]
        if 'experts.' in pname:
            expert_params += [p]
        elif 'linears' in pname:
            linear_params +=[p]

    params_id = list(map(id, expert_params)) + list(map(id, linear_params))
    base_parameters = list(filter(lambda p:id(p) not in params_id, model.parameters()))
    
    # optimizer = torch.optim.SGD([ {'params': base_parameters, 'lr': config['optimizer']['args']['share_lr']},
    #                               {'params': conv4_params, 'lr': config['optimizer']['args']['share_lr']},
    #                               {'params': linear_params, 'lr': config['optimizer']['args']['lr']}],
    #                               lr=config['optimizer']['args']['lr'],
    #                               momentum=config['optimizer']['args']['momentum'],
    #                               weight_decay=config['optimizer']['args']['weight_decay'],
    #                               nesterov=config['optimizer']['args']['nesterov'])
    optimizer = torch.optim.Adam([ {'params': base_parameters, 'lr': config['optimizer']['args']['share_lr']},
                                  {'params': expert_params, 'lr': config['optimizer']['args']['share_lr']},
                                  {'params': linear_params, 'lr': config['optimizer']['args']['lr']}],
                                  lr=config['optimizer']['args']['lr'],
                                 )

    lr_scheduler = learing_rate_scheduler(optimizer, config)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--dataset', default=None, type=str)
    args.add_argument('--imb_ratio', default=0.1, type=float)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--step1'], type=int, target='lr_scheduler;args;step1'),
        CustomArgs(['--step2'], type=int, target='lr_scheduler;args;step2'),
        CustomArgs(['--warmup'], type=int, target='lr_scheduler;args;warmup_epoch'),
        CustomArgs(['--gamma'], type=float, target='lr_scheduler;args;gamma'),
        CustomArgs(['--save_period'], type=int, target='trainer;save_period'),
        CustomArgs(['--reduce_dimension'], type=int, target='arch;args;reduce_dimension'),
        CustomArgs(['--layer2_dimension'], type=int, target='arch;args;layer2_output_dim'),
        CustomArgs(['--layer3_dimension'], type=int, target='arch;args;layer3_output_dim'),
        CustomArgs(['--layer4_dimension'], type=int, target='arch;args;layer4_output_dim'),
        CustomArgs(['--num_experts'], type=int, target='arch;args;num_experts') 
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
