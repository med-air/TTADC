import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset_synapse import Synapse_dataset, RandomGenerator
from test_func_TADE import inference
from  loss import focal_loss,prediction2label,ordinal_regression,ordinal_regression_focal
import torch.nn.functional as F
from tools import get_error_name
import time


def trainer_synapse(args, model, snapshot_path):
    
    logging.basicConfig(filename=snapshot_path + "log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split=args.train_txt,is_train = True,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train3")
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    class PadSequence:
        def __call__(self,batch):
            sorted_batch = sorted(batch,key=lambda x: x['image'].shape[0],reverse=True)
            sequences = [torch.from_numpy(x['image']) for x in sorted_batch]
            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences,batch_first=True)
            # sequences_padded = sequences_padded.to('cuda:1')
            # lengths = torch.LongTensor([len(x) for x in sequences])
            labels = [x['label'] for x in sorted_batch]
            labels = torch.tensor(np.array(labels))
            return sequences_padded, labels
    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    #                          worker_init_fn=worker_init_fn)
    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,worker_init_fn=worker_init_fn)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, collate_fn=PadSequence(),num_workers=4)


    device = torch.device("cuda:1")
    # torch.distributed.init_process_group(backend = "nccl")
    # torch.cuda.set_device('cude:{}'.format(args.device_ids[0]))
    model = torch.nn.DataParallel(model) 
    model = model.to(device)
    model = model.cuda()
    model.train()
    #criterion_phase = focal_loss(alpha=[4.14,7.89,7.6,19.0,2.24],gamma=2,num_classes=5)
    #criterion_phase = focal_loss(alpha=[2,2,2,2,1],gamma=2,num_classes=5)
    #criterion_phase = focal_loss(alpha=[2,4,4,4,1],gamma=2,num_classes=5)
    #criterion_phase = nn.CrossEntropyLoss(size_average=False)
    #criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([4.82,6.78,7.75,17.36,2.18])).float().cuda(),size_average=False)
    # criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([2.0,2.0,2.0,2.0,1.0])).float().cuda(),size_average=False)
    #criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([4.14,7.89,7.6,19.0,2.24])).float().cuda(),size_average=False)
    # optimizer = torch.optim.SGD([
    #             {'params': model.module.share.parameters(), },
    #             {'params': model.module.lstm.parameters(), 'lr': args.base_lr},
    #             {'params': model.module.fc1.parameters(), 'lr': args.base_lr},
    #             {'params': model.module.fc2.parameters(), 'lr': args.base_lr},
    #         ], lr = args.base_lr / 10,momentum = 0.9,weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.module.parameters(), lr = args.base_lr,momentum = 0.9,weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.module.parameters(), lr = args.base_lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 20,gamma=0.5)  
    error_name  = get_error_name(args.list_dir+args.val_txt+'.txt')
    # if args.model_step==0:
    #     error_name  = get_error_name(args.list_dir+args.val_txt+'.txt')
    # else:
    #     error_name = np.load('./results/error_name/'+args.val_txt+'.npy')
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs 
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(args.model_step,max_epoch+1), ncols=70)
    for epoch_num in iterator:
        error_name  = get_error_name(args.list_dir+args.val_txt+'.txt')
        #auc0,auc1,auc2,auc3,error_name = inference(args, model,error_name,epoch_num)
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch[0], sampled_batch[1]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            output = model(image_batch)
            # output = output.view(-1,length,5)
            # output = torch.mean(output,dim=1)
            # output = output.view(output.shape[0],5)
            # output = torch.sigmoid(output)
            # output_tran = prediction2label(output)
            loss = ordinal_regression_focal(output, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            
            
            # if iter_num % 20 == 0:
            #     image = image_batch[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

        
        save_interval = 50  # int(max_epoch/6)
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/total_loss', loss, iter_num)
        logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
        
        torch.cuda.empty_cache()
        time.sleep(10)
        
        auc0,auc1,auc2,auc3,error_name,result_,Y_val_set = inference(args, model,error_name,epoch_num)
        logging.info('epoch %d : auc0 : %f : auc1 : %f : auc2 : %f : auc3 : %f' % (epoch_num, auc0,auc1,auc2,auc3))
        model.train()
        print(args.model_path)
        if epoch_num % 1 ==0:
            print(error_name)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            save_result_path = os.path.join(snapshot_path, 'result_' + str(epoch_num) + '.npy')
            np.save(save_result_path,{'result':result_,'label':Y_val_set})
            torch.save(model.module.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.module.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
        np.save('./results/error_name/'+args.val_txt+'.npy',error_name)
        for num_name in error_name.keys():
            with open('./results/error_name/'+args.val_txt+'.txt','a') as fp:
                    fp.write(num_name+':'+str(error_name[num_name])+'\n')
                    fp.close
            

    writer.close()

    return "Training Finished!"
