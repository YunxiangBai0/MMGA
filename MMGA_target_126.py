import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from data.data_list import ImageList, ImageList_idx, ImageList_idx_aug, ImageList_idx_aug_fix
import random, pdb, math, copy
from tqdm import tqdm
import clip
import network, loss
from loss import IID_losses,info_loss
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from numpy import dtype, float32, linalg as LA
from data.data_loading import get_test_loader
from data.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK,IMAGENET_V_MASK
from net.model import *
from net import network,shot_model
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

def _get_gt_mask(logits, target):
    mask = torch.zeros_like(logits).scatter_(1, target, torch.ones(target.shape).cuda()).bool()
    mask_one = torch.zeros_like(logits).scatter_(1, target, torch.ones(target.shape).cuda())
    return mask,mask_one


def _get_other_mask(logits, target):
    mask = torch.ones_like(logits).scatter_(1, target, torch.zeros(target.shape).cuda()).bool()
    mask_one = torch.ones_like(logits).scatter_(1, target, torch.zeros(target.shape).cuda())
    return mask,mask_one

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask,gt_mask_one = _get_gt_mask(logits_student, target) #target index turn 1 [64,65]
    other_mask,other_mask_one = _get_other_mask(logits_student, target) 
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask) #[64,2],the first is target 
    
    logits_student_masked_5 = logits_student*gt_mask_one
    logits_student_masked_5 = F.softmax(logits_student_masked_5 / temperature, dim=1)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask) #[64,2]
    pred_teacher[:,0] = 1
    pred_teacher[:,1] = 0

    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss,logits_student_masked_5

def js_div(p_output, q_output, get_softmax=False):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
    dsets["target"] = ImageList_idx_aug_fix(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    return dset_loaders

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, base_model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[2]#90
            inputs = inputs.cuda()#(90,3,224,224)
            outputs = base_model(inputs)#(90,126)
            if start_test:
                all_output = outputs.float().cpu()#(30042,126)
                all_label = labels.float()#30042
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)#30042
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def train_target(args):
    text_inputs = clip_pre_text(args.FILE)
    if 'image' in args.dset:
        if args.net[0:3] == 'res':
            netF = network.ResBase(res_name=args.net)
        elif args.net[0:3] == 'vgg':
            netF = network.VGGBase(vgg_name=args.net)
        netC = network.Net2(2048,1000)
        base_model,transform = get_model(args, args.class_num)
        netC.linear.load_state_dict(base_model.fc.state_dict())
        del base_model
        Shot_model = shot_model.OfficeHome_Shot(netF,netC)
        # base_model = normalize_model(Shot_model, transform.mean, transform.std)
        base_model = Shot_model
        if args.dset == "imagenet_a":
            base_model = ImageNetXWrapper(base_model, IMAGENET_A_MASK)
        elif args.dset == "imagenet_r":
            base_model = ImageNetXWrapper(base_model, IMAGENET_R_MASK)
        elif args.dset == "imagenet_d109":
            base_model = ImageNetXWrapper(base_model, IMAGENET_D109_MASK)
        elif args.dset == "imagenet_v":
            base_model = ImageNetXWrapper(base_model, IMAGENET_V_MASK)
    else :
        base_model = get_model(args, args.class_num)


    base_model.cuda()
    param_group = []
    for k, v in base_model.named_parameters():
        if 'netC' in k or 'fc' in k:
            v.requires_grad = False
        else:
            param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    args.SEVERITY = [5]
    args.ADAPTATION = 'tent'
    args.NUM_EX = -1
    args.ALPHA_DIRICHLET = 0.0
    domain_name = args.type[args.t]
    dom_names_all = args.type

    target_data_loader = get_test_loader(setting=args.SETTING,
                                        adaptation=args.ADAPTATION,
                                        dataset_name=args.dset,
                                        root_dir=args.data,
                                        domain_name=domain_name,
                                        severity=args.level,
                                        num_examples=args.NUM_EX,
                                        rng_seed=args.seed,
                                        domain_names_all=dom_names_all,
                                        alpha_dirichlet=args.ALPHA_DIRICHLET,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        workers=args.worker,trans = image_train())


    test_data_loader = get_test_loader(setting=args.SETTING,
                                    adaptation=args.ADAPTATION,
                                    dataset_name=args.dset,
                                    root_dir=args.data,
                                    domain_name=domain_name,
                                    severity=args.level,
                                    num_examples=args.NUM_EX,
                                    rng_seed=args.seed,
                                    domain_names_all=dom_names_all,
                                    alpha_dirichlet=args.ALPHA_DIRICHLET,
                                    batch_size=args.batch_size*3,
                                    shuffle=False,
                                    workers=args.worker,trans = image_test())
    
    max_iter = args.max_epoch * len(target_data_loader)
    interval_iter = max_iter // args.interval
    iter_num = 0
    num_sample=len(target_data_loader.dataset)
    clip_list = []
    label_list = []

    while iter_num < max_iter:
        try:
            inputs_test, inputs_test_augs, target, tar_idx = next(iter_test)
        except:
            iter_test = iter(target_data_loader)
            inputs_test, inputs_test_augs, target, tar_idx = next(iter_test)
        if inputs_test.size(0) == 1:
            continue


        inputs_test = inputs_test.cuda()
        inputs_test_augs = inputs_test_augs.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        with torch.no_grad():
            if args.model_name == 'RN50':
                clip_features_aug = clip_image(inputs_test_augs)
            else: 
                clip_features_aug = clip_image(inputs_test)

        features_test = base_model.encoder(inputs_test) 

        clip_features = clip_features_aug
        clip_score,clip_textem = clip_text(text_inputs,clip_features)

        outputs_test = base_model.fc(features_test)   
        with torch.no_grad():
            new_clip = (outputs_test) + clip_score.cuda()

        _,clip_index = torch.max(new_clip, 1)


        postive = clip_textem[clip_index]
        loss_info_1 = info_loss.info_nce(features_test, postive) 
        loss_l1 = torch.nn.MSELoss(reduction='mean')
        loss_info_2 = loss_l1(features_test,clip_features)
        loss_info = 1*loss_info_1 + 1*loss_info_2
        
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        _,clip_index_5 = torch.topk(clip_score,args.class_n)
        _,clip_index_new = torch.max(new_clip, 1)
        clip_list.append(clip_index_new.cpu())
        label_list.append(target)

        dkl_loss,logits_student_masked_5 = dkd_loss(outputs_test, new_clip, clip_index_5, alpha=1.0, beta=0.0, temperature=1)

        clip_score_sm = nn.Softmax(dim=1)(new_clip)
        iic_loss = IID_losses.IID_loss(softmax_out, clip_score_sm)
        classifier_loss = 1.0 * iic_loss + args.cls_par* dkl_loss
        msoftmax = softmax_out.mean(dim=0)
        classifier_loss = classifier_loss + 1.0 * loss_info

        if  args.dset=='office':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
            classifier_loss = classifier_loss - 1.0 * gentropy_loss
        if  args.dset=='office-home':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
            classifier_loss = classifier_loss - 1.0 * gentropy_loss
        if  args.dset=='VISDA-C':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
            classifier_loss = classifier_loss - 0.1 * gentropy_loss
        if  args.dset=='domainnet126':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
            classifier_loss = classifier_loss - 0.5 * gentropy_loss
        
        """
        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss
        """

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            clip_list = torch.cat(clip_list)
            label_list = torch.cat(label_list)
            acc = (clip_list.int().cpu()==label_list.int()).float().mean().item()
            log_str = str(acc)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            base_model.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(test_data_loader, base_model, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(args.name, iter_num, max_iter, acc_s_te,classifier_loss) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(test_data_loader, base_model, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(args.name, iter_num, max_iter, acc_s_te,classifier_loss)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            base_model.train()
            clip_list = []
            label_list = []
    if args.issave:   
       torch.save(base_model.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
    
    return base_model


def print_args(args):
    s = "==========================================\n"    
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s



def clip_pre_text(FILE):
    List_rd = []
    with open(FILE) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    args.classname = classnames
    prompt_prefix = args.ctx_init.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts

def clip_text(text_inputs,image_features):
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.data
    logit_scale = logit_scale.exp().cpu()
    similarity = logit_scale * image_features @ text_features.T
    return similarity,text_features

def clip_pre(text_inputs,inputs_test):
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
        logits_per_image, logits_per_text = model(inputs_test, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return image_features,probs

def clip_image(inputs_test):
    image_input = inputs_test.cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features


if __name__ == "__main__":
    filename = os.path.basename(sys.argv[0])
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--filename', type=str, default=filename, help="filename")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1,help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='domainnet126', choices=['VISDA-C', 'office-home', 'domainnet126'])
    parser.add_argument('--lr', type=float, default=5e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.2)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--kd_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--bottleneck', type=int, default=1024)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='ckps/target_MMGA')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='RN50', choices=['RN50', 'ViT-B/32','RN101'])
    parser.add_argument('--class_n', type=int, default=3)
    parser.add_argument('--ctx_init', default='a_photo_of_a', type=str, help='init tunable prompts')
    parser.add_argument('--WEIGHTS',default='IMAGENET1K_V1', type=str)
    parser.add_argument('--SETTING',default='reset_each_shift', type=str)
    parser.add_argument('--level', default=5, type=int)
    parser.add_argument('--resume', default='source', help='directory of pretrained model')
    parser.add_argument('--data', default='/home/imi/data1/project/Datazoom',metavar='--DIR', help='path to dataset root')
    args = parser.parse_args()

    print("=======MMGA======")

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.FILE = './data/office-home/RealWorld_list_code2.txt'
        args.class_num = 65 
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.FILE = './data/VISDA-C/validation_list_code.txt'
        args.class_num = 12
    if args.dset == 'domainnet126':
        names = ["clipart", "painting", "real", "sketch"]
        args.FILE = './data/domainnet126/classname.txt'
        args.class_num = 126
        args.CKPT_PATH = osp.join(args.resume, args.da, args.dset + '_1024','best_' + names[args.s] +'_2020.pth.tar')
        args.bottleneck = 512

    args.type = names
    args.name_file = osp.join('./src/classname',args.dset,'classname.txt')
    args.names = names

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
    args.name = names[args.s][0].upper()+names[args.t][0].upper()
    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
    args.name = names[args.s][0].upper()+names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)
    if args.da == 'pda':
        args.gent = ''
        args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file_loss = open(osp.join(args.output_dir, 'log_loss' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    model, preprocess,_ = clip.load(args.model_name)
    model.float()
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False

    print("backbone:{}  dataset:{}\n".format(args.model_name, args.dset))
    train_target(args)