from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from model_queue import LDGnet
import numpy as np
from utils_HSI import sample_gt, metrics, get_device, seed_worker
from datasets import get_dataset, HyperX, data_prefetcher
from datetime import datetime
import os
import torch.utils.data as data
import scipy.io as io
from sklearn.metrics import classification_report
import clip
import time
parser = argparse.ArgumentParser(description='PyTorch LDGnet')

parser.add_argument('--save_path', type=str, default="/home/zyx/code/LDGnet/results/",
                    help='the path to save the model')
parser.add_argument('--data_path', type=str, default= '/home/zyx/data/Houston/',
                    help='the path to load the data')
# parser.add_argument('--save_path', type=str, default="./results/",
#                     help='the path to save the model')
# parser.add_argument('--data_path', type=str, default='./datasets/Houston/',
#                     help='the path to load the data')
parser.add_argument('--source_name', type=str, default='Houston13',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='Houston18',
                    help='the name of the test dir')
parser.add_argument('--cuda', type=int, default=1,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-2,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--lambda_1', type=float, default=1e+0,
                    help="Regularization parameter, balancing the alignment loss.")
group_train.add_argument('--alpha', type=float, default=0.3,
                    help="Regularization parameter, controlling the contribution of both coarse-and fine-grained linguistic features.")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--class_balancing', action='store_true',
                    help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int, default=256,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=3667, metavar='S',
                    help='random seed ')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')

parser.add_argument('--num_epoch', type=int, default=200,
                    help='the number of epoch')
parser.add_argument('--num_trials', type=int, default=1,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=5,
                    help='multiple of of data augmentation')

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',default=False,
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")

parser.add_argument('--with_exploration', default=True, action='store_true',
                    help="See data exploration visualization")

args = parser.parse_args()
DEVICE = get_device(args.cuda)

def train(epoch, model, num_epoch, label_name, label_queue):

    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / num_epoch), 0.75)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if (epoch-1)%10==0:
        print('learning rate{: .4f}'.format(LEARNING_RATE) )

    CNN_correct= 0
    iter_source = iter(train_loader)
    num_iter = len_src_loader

    for i in range(1, num_iter):

        model.train()
        data_src, label_src = iter_source.next()
        data_src, label_src = data_src.to(DEVICE), label_src.to(DEVICE)
        label_src = label_src - 1

        optimizer.zero_grad()
        text = torch.cat([clip.tokenize(f'A hyperspectral image of {label_name[k]}').to(k.device) for k in label_src])
        text_queue_1 = [label_queue[label_name[k]][0] for k in label_src]
        text_queue_2 = [label_queue[label_name[k]][1] for k in label_src]
        text_queue_1 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_1])
        text_queue_2 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_2])
        loss_coarse, loss_fine, label_src_pred = model(data_src, text, label_src, text_queue_1=text_queue_1, text_queue_2=text_queue_2)

        loss_cls = F.nll_loss(F.log_softmax(label_src_pred, dim=1), label_src.long())
        loss = loss_cls + args.lambda_1*((1-args.alpha)*loss_coarse + args.alpha*loss_fine)

        loss.backward()
        optimizer.step()

        pred = label_src_pred.data.max(1)[1]
        CNN_correct += pred.eq(label_src.data.view_as(pred)).cpu().sum()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format( epoch, i * len(data_src), len_src_dataset, 100. * i / len_src_loader))
            print('loss: {:.6f},  loss_cls: {:.6f},  loss_coarse: {:.6f}, loss_fine: {:.6f}'.format(
            loss.item(), loss_cls.item(), loss_coarse.item(), loss_fine.item()))
    CCN_acc = CNN_correct.item() / len_src_dataset
    print('[epoch: {:4}]  Train Accuracy: {:.4f} | train sample number: {:6}'.format(epoch, CCN_acc, len_src_dataset))

    return model, CCN_acc

def test(model, label_name):
    model.eval()
    loss = 0
    correct = 0
    loss_coarse = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            label = label - 1
            text = torch.cat([clip.tokenize(f'A hyperspectral image of {label_name[k]}').to(k.device) for k in label])
            loss_coarse_, label_src_pred = model(data, text, label)
            pred = label_src_pred.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(label_src_pred, dim = 1), label.long()).item()
            loss_coarse += loss_coarse_.item()
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss /= len_tar_loader
        loss_coarse /= len_tar_loader
        print('Average test loss: {:.4f}, loss clip: {:.4f}, test Accuracy: {}/{} ({:.2f}%), | test sample number: {:6}\n'.format(
            loss, loss_coarse, correct, len_tar_dataset, 100. * correct / len_tar_dataset, len_tar_dataset))
        
    return correct, correct.item() / len_tar_dataset, pred_list, label_list
if __name__ == '__main__':

    args.save_path = os.path.join(args.save_path)
    # args.save_path = os.path.join(args.save_path, args.source_name+'to'+args.target_name)
    acc_test_list, acc_maxval_test_list = np.zeros([args.num_trials,1]), np.zeros([args.num_trials,1])
    seed_worker(args.seed)

    img_src, gt_src, LABEL_VALUES_src, LABEL_QUEUE, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                            args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, LABEL_QUEUE, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                            args.data_path)
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])
    training_sample_tar_ratio = args.training_sample_ratio*args.re_ratio*sample_num_src/sample_num_tar

    num_classes=gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    r = int(hyperparams['patch_size']/2)+1
    img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric')
    img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
    gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))
    gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))     

    train_gt_src, _, training_set, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src

    for i in range(args.re_ratio-1): 
        img_src_con = np.concatenate((img_src_con,img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
    
    hyperparams_train = hyperparams.copy()
    hyperparams_train.update({'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})

    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    shuffle=True)
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                    pin_memory=True,
                                    # worker_init_fn=seed_worker,
                                    # generator=g,
                                    batch_size=hyperparams['batch_size'])                          
    len_src_loader = len(train_loader)
    len_src_dataset = len(train_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)

    print(hyperparams)
    print("train samples :",len_src_dataset)

    correct, acc = 0, 0
    pretrained_dict  = torch.load('./ViT-B-32.pt', map_location="cpu").state_dict()
    embed_dim = pretrained_dict ["text_projection"].shape[1]
    context_length = pretrained_dict ["positional_embedding"].shape[0]
    vocab_size = pretrained_dict ["token_embedding.weight"].shape[0]
    transformer_width = pretrained_dict ["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = 3
    model = LDGnet(embed_dim,
        img_src.shape[-1], hyperparams['patch_size'], gt_src.max(),
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers).to(DEVICE)
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in pretrained_dict:
            del pretrained_dict[key]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'visual' not in k.split('.')}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    # log_dir = os.path.join(args.save_path, time_str+'_lr_'+str(args.lr)+'_lam1_'+str(args.lambda_1)+'_alpha_'+str(args.alpha))
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    for epoch in range(1, args.num_epoch + 1):
        t1 = time.time()  
        model, CCN_train_acc = train(epoch, model, args.num_epoch, LABEL_VALUES_src, LABEL_QUEUE)
        t2 = time.time()
        print('epoch time:', t2-t1)

        t_correct, CCN_test_acc, pred, label = test(model, LABEL_VALUES_src)
        if t_correct > correct:
            correct = t_correct
            acc = CCN_test_acc
            results = metrics(np.concatenate(pred), np.concatenate(label), ignored_labels=hyperparams['ignored_labels'], n_classes=gt_src.max())
            print(classification_report(np.concatenate(pred),np.concatenate(label),target_names=LABEL_VALUES_tar))
            
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
            args.source_name, args.target_name, correct, 100. * correct / len_tar_dataset ))
        
        io.savemat(os.path.join(args.save_path, 'results_'+args.source_name+'_'+f'{CCN_test_acc*100 :.2f}'+'.mat'), 
                   {'lr':args.lr, 'lambda_1':args.lambda_1,'alpha':args.alpha,'results': results})