import torch
import numpy as np
from ARC import ARC
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import argparse
import os
import random
from meta import Meta
import pickle
from ARCDataset import ARCTrain
import matplotlib.pyplot as plt

seed = 42
print(f'setting everything to seed {seed}')
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('bn', [1]),
        ('conv2d', [64, 1, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [32, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('conv2d', [16, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [16]),
        ('conv2d', [16, 16, 3, 3, 1, 1]),
        ('relu', [True]),
        ('max_pool2d', [args.imgsz, args.imgsz, 0]),
        ('flatten', []),
        ('linear', [args.imgsz*args.imgsz*11, 16])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    arc = ARCTrain(root='/home/sid/Desktop/hertzwell/github/arc/data/',
               mode='train', n_way=1,
               k_shot=args.k_spt, k_query=1, batchsz=400, imgsz=args.imgsz)

    for epoch in range(args.epoch):
        # # fetch meta_batchsz num of episode each time
        # db = DataLoader(arc, args.task_num, shuffle=True,
        #                 num_workers=0, pin_memory=True)

        # for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(arc):
        all_accs, all_loss = [], []
        for step in range(len(arc)):
        # for step in range(2):

            x_spt, y_spt, x_qry, y_qry = arc[step]
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(
                device), x_qry.to(device), y_qry.to(device)

            accs, loss = maml(x_spt, y_spt, x_qry, y_qry)
            all_accs.append(accs)
            all_loss.append(loss)

            # if step % len(db)//5 == 0:
            print('epoch:', epoch, 'step:', step, '\ttraining acc:', accs)
        print()

        plt.subplot(1, 2, 1)
        plt.plot(np.mean(all_accs, 1), label=f'acc_epoch_{epoch}')
        plt.legend(loc="upper right")

        plt.subplot(1, 2, 2)
        plt.plot(all_loss, label=f'loss_epoch_{epoch}')
        plt.legend(loc="upper right")
        plt.show(block=False)
        plt.pause(0.1)
        print('epoch:', epoch, 'acc:', np.mean(all_accs, 0), 'loss:', np.mean(loss))
        print()
        if (epoch % 1 == 0) & (epoch > 0):  # evaluation
            # if (epoch % 10 == 0):  # evaluation
            with open(f'./model_weights/maml_sz30_epoch_{epoch}_accs_{np.mean(all_accs):.3}_loss_{np.mean(loss):.3}.pkl', 'wb') as f:
                pickle.dump(maml, f)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int,
                           help='epoch number', default=91)
    argparser.add_argument('--n_way', type=int, help='n way', default=1)
    argparser.add_argument('--k_spt', type=int,
                           help='k shot for support set', default=2)
    argparser.add_argument('--k_qry', type=int,
                           help='k shot for query set', default=1)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=15)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int,
                           help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float,
                           help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float,
                           help='task-level inner update learning rate', default=1)
    argparser.add_argument('--update_step', type=int,
                           help='task-level inner update steps', default=2)
    argparser.add_argument('--update_step_test', type=int,
                           help='update steps for finetunning', default=20)

    args = argparser.parse_args()

    main()
