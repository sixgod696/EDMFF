"""
Hello, welcome on board,
"""
from __future__ import print_function

import argparse
import os
import time, platform
import cv2
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING']="0"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from thop import profile
import torch.nn.utils.prune as prune
from dataset import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
from loss2 import *

from ted import TED # TEED architecture

from utils.img_processing import (image_normalization, save_image_batch_to_disk,
                   visualize_result, count_parameters)

is_testing =False
IS_LINUX = True if platform.system()=="Linux" else False

def train_one_epoch(epoch, dataloader, model, criterions, optimizer, device,
                    log_interval_vis, tb_writer, args=None):

    imgs_res_folder = os.path.join(args.output_dir, 'current_res')
    os.makedirs(imgs_res_folder,exist_ok=True)
    show_log = args.show_log
    if isinstance(criterions, list):
        criterion1, criterion2 = criterions
    else:
        criterion1 = criterions

    # Put model in training mode
    model.train()

    l_weight0 = [1.1,0.7,1.1,1.3] # for bdcn loss2-B4
    l_weight = [[0.05, 2.], [0.05, 2.], [0.01, 1.],
                [0.01, 3.]]  # for cats loss [0.01, 4.]
    loss_avg =[]
    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)  # BxCxHxW
        labels = sample_batched['labels'].to(device)  # BxHxW
        preds_list = model(images)
        loss1 = sum([criterion2(preds, labels,l_w) for preds, l_w in zip(preds_list[:-1],l_weight0)]) # bdcn_loss2 [1,2,3] TEED
        loss2 = criterion1(preds_list[-1], labels, l_weight[-1], device) # cats_loss [dfuse] TEED
        tLoss = loss2+loss1 # TEED

        optimizer.zero_grad()
        tLoss.backward()
        optimizer.step()
        loss_avg.append(tLoss.item())
        if epoch==0 and (batch_id==100 and tb_writer is not None):
            tmp_loss = np.array(loss_avg).mean()
            tb_writer.add_scalar('loss', tmp_loss,epoch)

        if batch_id % (show_log) == 0:
            print(time.ctime(), 'Epoch: {0} Sample {1}/{2} Loss: {3}'
                  .format(epoch, batch_id, len(dataloader), format(tLoss.item(),'.4f')))
        if batch_id % log_interval_vis == 0:
            res_data = []

            img = images.cpu().numpy()
            res_data.append(img[2])

            ed_gt = labels.cpu().numpy()
            res_data.append(ed_gt[2])

            # tmp_pred = tmp_preds[2,...]
            for i in range(len(preds_list)):
                tmp = preds_list[i]
                tmp = tmp[2]
                # print(tmp.shape)
                tmp = torch.sigmoid(tmp).unsqueeze(dim=0)
                tmp = tmp.cpu().detach().numpy()
                res_data.append(tmp)

            vis_imgs = visualize_result(res_data, arg=args)
            del tmp, res_data

            vis_imgs = cv2.resize(vis_imgs,
                                  (int(vis_imgs.shape[1]*0.8), int(vis_imgs.shape[0]*0.8)))
            img_test = 'Epoch: {0} Iter: {1}/{2} Loss: {3}' \
                .format(epoch, batch_id, len(dataloader), round(tLoss.item(),4))

            BLACK = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.9
            font_color = BLACK
            font_thickness = 2
            x, y = 30, 30
            vis_imgs = cv2.putText(vis_imgs,
                                   img_test,
                                   (x, y),
                                   font, font_size, font_color, font_thickness, cv2.LINE_AA)
            # tmp_vis_name = str(batch_id)+'-results.png'
            # cv2.imwrite(os.path.join(imgs_res_folder, tmp_vis_name), vis_imgs)
            cv2.imwrite(os.path.join(imgs_res_folder, 'results.png'), vis_imgs)
    loss_avg = np.array(loss_avg).mean()
    return loss_avg

def validate_one_epoch(epoch, dataloader, model, device, output_dir, arg=None,test_resize=False):
    # XXX This is not really validation, but testing

    # Put model in eval mode
    model.eval()

    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            # labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            preds = model(images,single_test=test_resize)
            # print('pred shape', preds[0].shape)
            save_image_batch_to_disk(preds[-1],
                                     output_dir,
                                     file_names,img_shape=image_shape,
                                     arg=arg)


def test(checkpoint_path, dataloader, model, device, output_dir, args,resize_input=False):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    model.eval()
    # just for the new dataset
    # os.makedirs(os.path.join(output_dir,"healthy"), exist_ok=True)
    # os.makedirs(os.path.join(output_dir,"infected"), exist_ok=True)

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            # if not args.test_data == "CLASSIC":
            labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']


            print(f"{file_names}: {images.shape}")
            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            preds = model(images, single_test=resize_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)
            save_image_batch_to_disk(preds,
                                     output_dir, # output_dir
                                     file_names,
                                     image_shape,
                                     arg=args)
            torch.cuda.empty_cache()
    total_duration = np.sum(np.array(total_duration))

    print("******** Testing finished in", args.test_data, "dataset. *****")
    print(total_duration)
    print(len(dataloader))


    print("FPS: %f.4" % (len(dataloader)/total_duration))
    # print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")

def testPich(checkpoint_path, dataloader, model, device, output_dir, args, resize_input=False):
    # a test model plus the interganged channels
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            start_time = time.time()
            images2 = images[:, [1, 0, 2], :, :]  #GBR
            # images2 = images[:, [2, 1, 0], :, :] # RGB
            preds = model(images,single_test=resize_input)
            preds2 = model(images2,single_test=resize_input)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            save_image_batch_to_disk([preds,preds2],
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args, is_inchannel=True)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")

def parse_args(is_testing=True):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TEED model')
    parser.add_argument('--choose_test_data',
                        type=int,
                        default=15,     # UDED=15
                        help='Choose a dataset for testing: 0 - 15')

    # ----------- test -------0--
    TEST_DATA = DATASET_NAMES[parser.parse_args().choose_test_data] # max partialnet
    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)

    # Training settings
    # BIPED-B2=1, BIPDE-B3=2, just for evaluation, using LDC trained with 2 or 3 bloacks
    TRAIN_DATA = DATASET_NAMES[0] # BIPED=0, BRIND=6, MDBD=10, BIPBRI=13
    train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)
    train_dir = train_inf['data_dir']

    # Data parameters
    parser.add_argument('--input_dir',
                        type=str,
                        default=train_dir,
                        help='the path to the directory with the input disease_data.')
    parser.add_argument('--input_val_dir',
                        type=str,
                        default=test_inf['data_dir'],
                        help='the path to the directory with the input disease_data for validation.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='checkpoints',
                        help='the path to output the results.')
    parser.add_argument('--train_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TRAIN_DATA,
                        help='Name of the dataset.')# TRAIN_DATA,BIPED-B3
    parser.add_argument('--test_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TEST_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--test_list',
                        type=str,
                        default=test_inf['test_list'],
                        help='Dataset sample indices list.')
    parser.add_argument('--train_list',
                        type=str,
                        default=train_inf['train_list'],
                        help='Dataset sample indices list.')
    parser.add_argument('--is_testing',type=bool,
                        default=is_testing,
                        help='Script in testing mode.')
    parser.add_argument('--predict_all',
                        type=bool,
                        default=False,
                        help='True: Generate all TEED outputs in all_edges ')
    parser.add_argument('--up_scale',
                        type=bool,
                        default=False, # for Upsale test set in 30%
                        help='True: up scale x1.5 test image')  # Just for test

    parser.add_argument('--resume',
                        type=bool,
                        default=False,#修改，是否resume已有的模型
                        help='use previous trained disease_data')  # Just for test
    parser.add_argument('--checkpoint_data',
                        type=str,
                        default='100/100_model.pth',#修改 test测试 测试模型选择 37 for biped 60 MDBD  partialnet/8_model.pth
                        help='Checkpoint path.')
    parser.add_argument('--test_img_width',
                        type=int,
                        default=test_inf['img_width'],
                        help='Image width for testing.')
    parser.add_argument('--test_img_height',
                        type=int,
                        default=test_inf['img_height'],
                        help='Image height for testing.')
    parser.add_argument('--res_dir',
                        type=str,
                        default='result',
                        help='Result directory')
    parser.add_argument('--use_gpu',type=int,
                        default=0, help='use GPU')
    parser.add_argument('--log_interval_vis',
                        type=int,
                        default=500,# 100
                        help='Interval to visualize predictions. 200')
    parser.add_argument('--show_log', type=int, default=200, help='display logs')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        metavar='N',
                        help='Number of training epochs (default: 25).')#修改epoch
    parser.add_argument('--lr', default=8e-4, type=float,
                        help='Initial learning rate. =1e-3') # 1e-3
    parser.add_argument('--lrs', default=[9e-5], type=float,
                        help='LR for epochs') #修改，原本应该是8e-5  [7e-5]
    parser.add_argument('--wd', type=float, default=2e-4, metavar='WD',
                        help='weight decay (Good 5e-4/1e-4  )') # good 12e-5
    parser.add_argument('--adjust_lr', default=[4], type=int,
                        help='Learning rate step size.')  # [4] [6,9,19]
    parser.add_argument('--version_notes',
                        default='TEED BIPED+BRIND-trainingdataLoader BRIND light AF -USNet--noBN xav init normal bdcnLoss2+cats2loss +DoubleFusion-3AF, AF sum',
                        type=str,
                        help='version notes')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        metavar='B',
                        help='the mini-batch size (default: partialnet)')
    parser.add_argument('--workers',
                        default=0,#修改workers
                        type=int,
                        help='The number of workers for the dataloaders.')
    parser.add_argument('--tensorboard',type=bool,
                        default=True,
                        help='Use Tensorboard for logging.'),
    parser.add_argument('--img_width',
                        type=int,
                        default=300,
                        help='Image width for training.') # BIPED 352/300 BRIND 256 MDBD 480
    parser.add_argument('--img_height',
                        type=int,
                        default=300,
                        help='Image height for training.') # BIPED 352/300 BSDS 352/320
    parser.add_argument('--channel_swap',
                        default=[2, 1, 0],
                        type=int)
    parser.add_argument('--resume_chpt',
                        default='result/resume/',
                        type=str,
                        help='resume training')
    parser.add_argument('--crop_img',
                        default=True,
                        type=bool,
                        help='If true crop training images, else resize images to match image width and height.')
    parser.add_argument('--mean_test',
                        default=test_inf['mean'],
                        type=float)
    parser.add_argument('--mean_train',
                        default=train_inf['mean'],
                        type=float)  # [103.939,116.779,123.68,137.86] [104.00699, 116.66877, 122.67892]

    args = parser.parse_args()
    return args, train_inf


def main(args, train_inf):

    # Tensorboard summary writer

    # torch.autograd.set_detect_anomaly(True)
    tb_writer = None
    training_dir = os.path.join(args.output_dir,args.train_data)
    os.makedirs(training_dir,exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, args.train_data,args.checkpoint_data)
    
    print("----------------")
    print(checkpoint_path)
    print("----------------")
    
    
    if args.tensorboard and not args.is_testing:
        # from tensorboardX import SummaryWriter  # previous torch version
        from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
        tb_writer = SummaryWriter(log_dir=training_dir)
        # saving training settings
        training_notes =[args.version_notes+ ' RL= ' + str(args.lr) + ' WD= '
                          + str(args.wd) + ' image size = ' + str(args.img_width)
                          + ' adjust LR=' + str(args.adjust_lr) +' LRs= '
                          + str(args.lrs)+' Loss Function= BDCNloss2 + CAST-loss2.py '
                          + str(time.asctime())+' trained on '+args.train_data]
        info_txt = open(os.path.join(training_dir, 'training_settings.txt'), 'w')
        info_txt.write(str(training_notes))
        info_txt.close()
        print("Training details> ",training_notes)

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    # torch.cuda.set_device(args.use_gpu) # set a desired gpu

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")
    # print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Trainimage mean: {args.mean_train}')
    print(f'Test image mean: {args.mean_test}')


    # Instantiate model and move it to the computing device
    model = TED().to(device)
    # model = nn.DataParallel(model)
    ini_epoch =0
    if not args.is_testing:
        if args.resume:
            checkpoint_path2= os.path.join(args.output_dir, 'BIPED-54-B4',args.checkpoint_data)
            ini_epoch=12
            model.load_state_dict(torch.load(checkpoint_path2,
                                         map_location=device))

        # Dataset Loading...
        dataset_train = BipedDataset(args.input_dir,
                                     img_width=args.img_width,
                                     img_height=args.img_height,
                                     train_mode='train',
                                     arg=args
                                     )
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                  num_workers=args.workers)
    # Test dataset loading
    dataset_val = TestDataset(args.input_val_dir,
                              test_data=args.test_data,
                              img_width=args.test_img_width,
                              img_height=args.test_img_height,
                              test_list=args.test_list, arg=args
                              )
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)
    # Testing
    if_resize_img = False if args.test_data in ['BIPED', 'CID', 'MDBD'] else True
    if args.is_testing:
        # ##剪枝部分，训练完毕才可以进行剪枝，修改部分
        # # 在训练后，选择要剪枝的层，例如第一个卷积层
        # co1 = model.block_lxt.conv1
        # co2 = model.block_1.conv1
        # # 应用剪枝，例如移除权重的20%
        # prune.l1_unstructured(co1, name="weight", amount=0.2)
        # prune.l1_unstructured(co2, name="weight", amount=0.2)
        # # 可选: 在剪枝后将剪枝应用到模型，并删除剪枝掩码以节省空间
        # prune.remove(co1, 'weight')
        # prune.remove(co2, 'weight')

        output_dir = os.path.join(args.res_dir, args.train_data+"2"+ args.test_data)
        print(f"output_dir: {output_dir}")

        test(checkpoint_path, dataloader_val, model, device,
             output_dir, args,if_resize_img)

        # Count parameters:
        num_param = count_parameters(model)
        print('-------------------------------------------------------')
        print('lxt test parameters:')
        print(num_param)
        print('-------------------------------------------------------')
        return

    criterion1 = cats_loss #bdcn_loss2
    criterion2 = bdcn_loss2#cats_loss#f1_accuracy2
    criterion = [criterion1,criterion2]
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)

    # Count parameters:
    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('LXT parameters:')
    print(num_param)
    print( " ***** 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 *****")
    print('-------------------------------------------------------')

    # Main training loop
    seed=1021
    adjust_lr = args.adjust_lr
    k=0
    set_lr = args.lrs#[25e-4, 5e-6]
    for epoch in range(ini_epoch,args.epochs):
        if epoch%5==0: # before 7

            seed = seed+1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("------ Random seed applied-------------")
        # adjust learning rate
        if adjust_lr is not None:
            if epoch in adjust_lr:
                lr2 = set_lr[k]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr2
                k+=1
        # Create output directories

        output_dir_epoch = os.path.join(args.output_dir,args.train_data, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch, args.test_data + '_res')
        os.makedirs(output_dir_epoch,exist_ok=True)
        os.makedirs(img_test_dir,exist_ok=True)
        print("**************** Validating the training from the scratch **********")
        # validate_one_epoch(epoch,
        #                    dataloader_val,
        #                    model,
        #                    device,
        #                    img_test_dir,
        #                    arg=args,test_resize=if_resize_img)

        avg_loss =train_one_epoch(epoch,dataloader_train,
                        model, criterion,
                        optimizer,
                        device,
                        args.log_interval_vis,
                        tb_writer=tb_writer,
                        args=args)
        validate_one_epoch(epoch,
                           dataloader_val,
                           model,
                           device,
                           img_test_dir,
                           arg=args, test_resize=if_resize_img)





        # Save model after end of every epoch
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                   os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))
        if tb_writer is not None:
            tb_writer.add_scalar('loss',
                                 avg_loss,
                                 epoch+1)
        # print("--------ceshi---------")
        # print(tb_writer)
        # print("--------ceshi---------")

        print('Last learning rate> ', optimizer.param_groups[0]['lr'])

    # ##剪枝部分，训练完毕才可以进行剪枝，修改部分
    # # 在训练后，选择要剪枝的层，例如第一个卷积层
    # conv1 = model.block_lxt[0]
    # conv2 = model.block_1[0]
    # # 应用剪枝，例如移除权重的20%
    # prune.l1_unstructured(conv1, name="weight", amount=0.2)
    # prune.l1_unstructured(conv2, name="weight", amount=0.2)
    # # 可选: 在剪枝后将剪枝应用到模型，并删除剪枝掩码以节省空间
    # prune.remove(conv1, 'weight')
    # prune.remove(conv2, 'weight')

    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('TEED parameters:')
    print(num_param)
    print('-------------------------------------------------------')

if __name__ == '__main__':
    # os.system(" ".join(command))
    is_testing = False# True to use TEED for testing
    args, train_info = parse_args(is_testing=is_testing)
    main(args, train_info)

# python main.py --choose_test_data=-1