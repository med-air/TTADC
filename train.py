import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from network import *
from trainer import trainer_synapse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


parser = argparse.ArgumentParser()


parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--list_dir', type=str,
                    default='./lists/', help='list dir')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--max_epochs', type=int,
                    default=45, help='maximum epoch number to train')
parser.add_argument('--root_path', type=str,
                    default='../data/final/iCTCF_6_24', help='root dir for data')
parser.add_argument('--model_path', type=str,
                    default='../models_save/iCTCF_6_test_bs4_5e5_de20_40_/')
parser.add_argument('--model_step', type=str,
                    default= 40)
parser.add_argument('--max_step', type=str,
                    default= 30000)
parser.add_argument('--batch_size', type=str,
                    default= 20)
parser.add_argument('--base_lr', type=str,
                    default= 5e-5)
parser.add_argument('--sequence_length', type=str,
                    default= 11)
parser.add_argument('--train_txt', type=str,
                    default= 'iCTCF_train')
parser.add_argument('--val_txt', type=str,
                    default= 'iCTCF_test')
parser.add_argument('--num_classes', type=int,
                    default=6, help='output channel of network')

args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # dataset_name = args.dataset
    # dataset_config = {
    #     'Synapse': {
    #         'root_path': '../data/final',
    #         'list_dir': './lists/lists_Synapse',
    #         'num_classes': 2,
    #     },
    # }
    # args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.root_path = dataset_config[dataset_name]['root_path']
    # args.list_dir = dataset_config[dataset_name]['list_dir']
    # args.is_pretrain = True
    # args.exp = 'TU_' + dataset_name + str(args.img_size)
    # snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    # snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    # snapshot_path += '_' + args.vit_name
    # snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    # snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    # snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    # snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    # snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    # snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    # snapshot_path = snapshot_path + '_'+str(args.img_size)
    # snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    model = resnet_lstm4_fle_TADE_light()
    if args.model_step != 0:
        model_path = args.model_path  +'epoch_'+ str(args.model_step) + '.pth'
        model.load_state_dict(torch.load(model_path))


    trainer_synapse(args, model, args.model_path)
