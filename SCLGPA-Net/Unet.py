import argparse
import os
import random
import numpy as np
from itertools import cycle
from tqdm import tqdm
import wandb
import warnings
import logging
warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
torch.backends.cudnn.enabled = False

from utils import losses, data_loading
from functions import create_model
from evaluate import evaluate
from transformers import transformer_img, val_form,height, width
from random_val_test import split_val_test
import os
os.environ["WANDB_MODE"] = "dryrun"
def save_model(args, global_step, model):
    save_mode_path = os.path.join(args.save_path, '_best_' +'_Unet_'+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    print(f'Checkpoint {global_step} saved!')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='./train',
                        help='the path of training data')

    parser.add_argument('--val_path',
                        type=str,
                        default='./val',
                        help='the path of val data')
    
    parser.add_argument('--test_path',
                        type=str,
                        default='./test',
                        help='the path of val data')
    parser.add_argument('--save_path',
                        type=str,
                        default='./checkpoints',
                        help='the path of save_model')
    parser.add_argument('--deterministic',
                        type=int,
                        default=0,
                        help='whether use deterministic training')
    parser.add_argument('--epochs',
                        type=int,
                        default=300,
                        help='maximum iterations number to train')

    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-5,
                        help='maximum epoch number to train')
    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='the batch_size of training size')
    parser.add_argument('--amp',
                        action='store_true',
                        default=False,
                        help='Use mixed precision')
 

    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_parser()
    split_val_test()
    labeled_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    amp = args.amp
    num_classes = 4
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')
    labeled_dataset = data_loading.BasicDataset(
        imgs_dir=labeled_path + '/' + 'imgs/',
        masks_dir=labeled_path + '/' + 'masks/',
        augumentation=transformer_img())
    n_train = len(labeled_dataset)

    val_dataset = data_loading.BasicDataset(
        imgs_dir=val_path + '/' + 'imgs/',
        masks_dir=val_path + '/' + 'masks/',
        augumentation=val_form())
    
    test_dataset = data_loading.BasicDataset(
        imgs_dir=test_path + '/' + 'imgs/',
        masks_dir=test_path + '/' + 'masks/',
        augumentation=val_form())
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    labeled_dataloader = DataLoader(dataset=labeled_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True,
                                    worker_init_fn=worker_init_fn)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)
    logging.info(f'''Starting training:
    Epochs:          {epochs}
    Batch size(labeled):      {batch_size}
    Learning rate:   {learning_rate}
    Training size:   {height, width}
    Checkpoints:     {args.save_path}
    Device:          {device.type}
    Mixed Precision: {amp}
    ''')
    model = create_model(device=device,num_classes=num_classes)
    # model.load_state_dict(torch.load(r'C:\Users\10194\Desktop\Unet_effusion_segmentation\checkpoints\test_100.pth', map_location=torch.device('cuda')))

    model.train()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    CEloss = torch.nn.CrossEntropyLoss(reduction='none')
    global_step = 0
    max_iou=0
    for epoch in range(1, epochs+1):
        progress_bar = tqdm(labeled_dataloader, desc="Epoch {} - {}".format(epoch, 'training'))
        losses_unet = []
        iter_num = 0
        model.train()
        for sampled_batch in progress_bar:
                #labeled data and weak labeded data
            labeled_imgs, labeled_masks = sampled_batch['image'], sampled_batch['mask']
            labeled_imgs, labeled_masks = labeled_imgs.to(device=device, dtype=torch.float32), labeled_masks.to(
                        device=device, dtype=torch.long)

            outputs_labeled = model(labeled_imgs)
                    #计算有监督损失
            supervised_loss = torch.mean(CEloss(outputs_labeled, labeled_masks))

            loss_unet = supervised_loss

            optimizer.zero_grad()
            loss_unet.backward()
            optimizer.step()
            losses_unet.append(loss_unet.item())

            iter_num = iter_num + 1

            progress_bar.set_postfix(loss_unet=np.mean(losses_unet))
		
            if global_step != 0 and global_step % 100 == 0:
                val_P, val_S, val_I, asd_score, hd95_score, m_iou, std_iou = evaluate(model, val_dataloader, device, num_classes)
                val_dice = (val_P + val_S + val_I) / 3
                if m_iou > max_iou:
                    save_model(args,str(global_step), model)#save best model
                    max_iou = m_iou
                logging.info('Validation Dice score: {},{},{},{},{}'.format(val_P, val_S, val_I, val_dice, m_iou))

            if global_step % 2000 == 0:
                lr = learning_rate * 0.1 ** (global_step // 2000)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if global_step >=4000:
                break
            global_step += 1
        if global_step >=40000:
            break
    model_test = create_model(device=device,num_classes=num_classes)
    model_test.load_state_dict(torch.load('checkpoints/_best__Unet_.pth', map_location='cpu'))
    model_test.to(device)
    test_P, test_S, test_I, asd_score, hd95_score,m_iou,std_iou = evaluate(model_test, test_dataloader, device, num_classes)
    test_dice = (test_P + test_S + test_I) / 3
    with open('Unet.txt', 'a') as file:
        # 写入文本内容，这将添加到文件的末尾
        file.write('test Iou score: {},{}\n'.format(round(m_iou, 4),round(std_iou,4)))
    logging.info('test Iou score: {},{}\n'.format(round(m_iou, 4),round(std_iou,4)))


    