import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester


parser = argparse.ArgumentParser(description='implementation of MonoLSS')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--config', type=str, default='lib/kitti.yaml')
args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def main():   
    #assert (torch.cuda.is_available())

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


    # load cfg
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    os.makedirs(cfg['trainer']['log_dir'], exist_ok=True)
    logger = create_logger(os.path.join(cfg['trainer']['log_dir'], 'train.log'))

    import shutil
    if not args.evaluate:
        if not args.test:
            if os.path.exists(os.path.join(cfg['trainer']['log_dir'], 'lib/')):
                shutil.rmtree(os.path.join(cfg['trainer']['log_dir'], 'lib/'))
        if not args.test:
            shutil.copytree('./lib', os.path.join(cfg['trainer']['log_dir'], 'lib/'))
        
    
    #  build dataloader
    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model = build_model(cfg['model'], train_loader.dataset.cls_mean_size)

    # evaluation mode
    if args.evaluate:
        print('evaluation mode')
        tester = Tester(cfg['tester'], cfg['dataset'], model, val_loader, logger)
        tester.test()
        return

    if args.test:
        print('test mode')
        tester = Tester(cfg['tester'], cfg['dataset'], model, test_loader, logger)
        tester.test()
        return

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr & bnm scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg,
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger)
    trainer.train()


if __name__ == '__main__':
    main()
