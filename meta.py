import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np

from learner import Learner
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import colors
from focal_loss import FocalLoss
from scipy import ndimage


cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF'])
norm = colors.Normalize(vmin=0, vmax=10)
device = 'cuda'


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

        tmp = filter(lambda x: x.requires_grad, self.net.parameters())
        self.num_weights = sum(map(lambda x: np.prod(x.shape), tmp))
        layer_names = list(map(lambda x: x[0], config))
        # initialise meta-parameters
        # self.theta = {}
        # self.theta['mean'] = {}
        # self.theta['logSigma'] = {}
        # for key, var in zip(layer_names, self.net.vars):
        #     w_target_shape = var.shape
        #     # if 'b' in key:
        #     print(key, w_target_shape)
        #     if w_target_shape == (1,) or len(w_target_shape) < 2:
        #         self.theta['mean'][key] = torch.zeros(w_target_shape, device=device, requires_grad=True)
        #     else:
        #         self.theta['mean'][key] = torch.empty(w_target_shape, device=device)
        #         torch.nn.init.xavier_normal_(tensor=self.theta['mean'][key], gain=1.)
        #         self.theta['mean'][key].requires_grad_()
        #     self.theta['logSigma'][key] = torch.rand(w_target_shape, device=device) - 4
        #     self.theta['logSigma'][key].requires_grad_()

        self.theta = {}
        self.theta['mean'] = {}
        self.theta['logSigma'] = {}
        for key, var in enumerate(self.net.vars):
            w_target_shape = var.shape
            # if 'b' in key:
            print(key, w_target_shape)
            if w_target_shape == (1,) or len(w_target_shape) < 2:
                self.theta['mean'][key] = torch.zeros(
                    w_target_shape, device=device, requires_grad=True)
            else:
                self.theta['mean'][key] = torch.empty(
                    w_target_shape, device=device)
                torch.nn.init.xavier_normal_(
                    tensor=self.theta['mean'][key], gain=1.)
                self.theta['mean'][key].requires_grad_()
            self.theta['logSigma'][key] = torch.rand(
                w_target_shape, device=device) - 4
            self.theta['logSigma'][key].requires_grad_()

        meta_lr = 1e-3
        self.op_theta = torch.optim.Adam(
            [
                {
                    'params': self.theta['mean'].values(),
                    'weight_decay': 0
                },
                {
                    'params': self.theta['logSigma'].values()
                }
            ],
            lr=meta_lr
        )

        # self.loss_fn = FocalLoss(num_class=11, gamma=2)
        self.loss_fn = F.cross_entropy

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        setsz, c_, h, w = x_spt.size()

        L, Lv = 32, 16
        inner_lr = 1e-3

        # losses_q[i] is the loss on step i
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        q = initialise_dict_of_dict(self.theta['mean'].keys())

        loss_vfe = 0
        for _ in range(L):
            w = generate_weights(meta_params=self.theta)
            # for i in range(setsz):
            y_pred_t = self.net.forward(x=x_spt, vars=w).reshape(-1, 11)
            # loss_t = loss_fn(y_pred_t, y_t)
            loss_vfe = loss_vfe + self.loss_fn(y_pred_t, y_spt)

        if y_qry is not None:
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(
                    x_qry, w, bn_training=True).reshape(-1, 11)
                loss_q = F.cross_entropy(logits_q, y_qry)
                # loss_q = self.loss_fn(logits_q, y_qry[0])
                losses_q[0] += loss_q.item()

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[0] = corrects[0] + correct

        loss_vfe = loss_vfe/setsz
        grads_mean = torch.autograd.grad(
            outputs=loss_vfe,
            inputs=self.theta['mean'].values(),
            create_graph=True
        )
        grads_logSigma = torch.autograd.grad(
            outputs=loss_vfe,
            inputs=self.theta['logSigma'].values(),
            create_graph=True
        )
        gradients_mean = dict(zip(self.theta['mean'].keys(), grads_mean))
        gradients_logSigma = dict(
            zip(self.theta['logSigma'].keys(), grads_logSigma))

        for key in w.keys():
            q['mean'][key] = self.theta['mean'][key] - \
                inner_lr*gradients_mean[key]
            q['logSigma'][key] = self.theta['logSigma'][key] - \
                inner_lr*gradients_logSigma[key]

        num_inner_updates = 5
        '''2nd update'''
        for _ in range(num_inner_updates - 1):
            loss_vfe = 0
            for _ in range(L):
                w = generate_weights(meta_params=q)
                y_pred_t = self.net.forward(x=x_spt, vars=w).reshape(-1, 11)
                loss_vfe = loss_vfe + self.loss_fn(y_pred_t, y_spt)
            loss_vfe = loss_vfe/L
            grads_mean = torch.autograd.grad(
                outputs=loss_vfe,
                inputs=q['mean'].values(),
                retain_graph=True
            )
            grads_logSigma = torch.autograd.grad(
                outputs=loss_vfe,
                inputs=q['logSigma'].values(),
                retain_graph=True
            )
            gradients_mean = dict(zip(q['mean'].keys(), grads_mean))
            gradients_logSigma = dict(
                zip(q['logSigma'].keys(), grads_logSigma))

            KL_reweight = 1
            for key in w.keys():
                q['mean'][key] = q['mean'][key] - inner_lr*(gradients_mean[key]
                                                            - KL_reweight*torch.exp(-2*self.theta['logSigma'][key])*(self.theta['mean'][key] - q['mean'][key]))
                q['logSigma'][key] = q['logSigma'][key] - inner_lr*(gradients_logSigma[key]
                                                                    + KL_reweight*(torch.exp(2*(q['logSigma'][key] - self.theta['logSigma'][key])) - 1))

        if y_qry is not None:
            # this is the loss and accuracy after first update
            with torch.no_grad():
                w = generate_weights(meta_params=q)
                # [setsz, nway]
                logits_q = self.net(
                    x_qry, w, bn_training=True).reshape(-1, 11)
                loss_q = F.cross_entropy(logits_q, y_qry)
                # loss_q = self.loss_fn(logits_q, y_qry[0])
                losses_q[1] += loss_q.item()

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[1] = corrects[1] + correct
        
            '''Task prediction'''
            loss_NLL = 0
            for _ in range(Lv):
                w = generate_weights(meta_params=q)
                y_pred_ = self.net.forward(x=torch.cat((x_spt, x_qry)), vars=w).reshape(-1, 11)
                loss_NLL = loss_NLL + self.loss_fn(y_pred_, torch.cat((y_spt, y_qry)))

            KL_loss = 0
            for key in q['mean'].keys():
                KL_loss += torch.sum(torch.exp(2*(q['logSigma'][key] - self.theta['logSigma'][key]))
                                    + (self.theta['mean'][key] - q['mean'][key])**2/torch.exp(2*self.theta['logSigma'][key]))\
                    + torch.sum(2*(self.theta['logSigma']
                                [key] - q['logSigma'][key]))
            KL_loss = (KL_loss - self.num_weights)/2

            # accumulate the loss of many ensambling networks to descent gradient for meta update
            # average over the number of tasks per minibatch
            loss_NLL = loss_NLL/Lv/setsz
            self.op_theta.zero_grad()
            loss_NLL.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(parameters=theta.values(), max_norm=1)
            self.op_theta.step()

            print('kl_loss:', f'{KL_loss.item():.5}',
                'loss_nll:', f'{loss_NLL.item():.3}')

        if y_qry is not None:
            # this is the loss and accuracy after first update
            with torch.no_grad():
                w = generate_weights(meta_params=self.theta)
                # [setsz, nway]
                logits_q = self.net(
                    x_qry, w, bn_training=True).reshape(-1, 11)
                loss_q = F.cross_entropy(logits_q, y_qry)
                # loss_q = self.loss_fn(logits_q, y_qry[0])
                losses_q[2] += loss_q.item()

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[2] = corrects[2] + correct
        else:
            with torch.no_grad():
                w = generate_weights(meta_params=self.theta)
                # [setsz, nway]
                logits_q = self.net(
                    x_qry, w, bn_training=True).reshape(-1, 11)
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                return pred_q
        # accs = np.array(corrects) / (querysz * setsz)
        accs = np.array(corrects) / (setsz * pred_q.shape[0])

        return accs, np.mean(losses_q)

    def finetunning(self, x_spt, y_spt, x_qry, im_num):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt).reshape(-1, 11)
        print('first iter', (logits.argmax(1) ==
                             y_spt).sum().item()/y_spt.shape[0])
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(),
                           bn_training=True).reshape(-1, 11)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            # correct = torch.eq(pred_q, y_qry).sum().item()
            # corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights,
                           bn_training=True).reshape(-1, 11)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            # correct = torch.eq(pred_q, y_qry).sum().item()
            # corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True).reshape(-1, 11)
            loss = F.cross_entropy(logits, y_spt)
            # print('k=',k, (logits.argmax(1)==y_spt).sum().item()/y_spt.shape[0])

            # 2. compute grad on self.theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. self.theta_pi = self.theta_pi - train_lr * grad
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights,
                           bn_training=True).reshape(-1, 11)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            # loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                # # convert to numpy
                # correct = torch.eq(pred_q, y_qry).sum().item()
                # corrects[k + 1] = corrects[k + 1] + correct

        del net
        img_sz = 30
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(x_spt[0].cpu().numpy().reshape(img_sz, img_sz),
                   cmap=cmap, norm=norm)
        plt.subplot(2, 2, 2)
        plt.imshow(y_spt[:img_sz*img_sz].cpu().numpy().reshape(img_sz, img_sz),
                   cmap=cmap, norm=norm)

        plt.subplot(2, 2, 3)
        plt.imshow(x_qry[0].cpu().numpy().reshape(img_sz, img_sz),
                   cmap=cmap, norm=norm)

        pred_q = pred_q[:img_sz*img_sz].cpu().numpy().reshape(img_sz, img_sz)
        # print(zip(*np.where(pred_q == 1)))
        frow = np.nonzero(np.count_nonzero(pred_q-10, axis=1))[0][0]
        fcol = np.nonzero(np.count_nonzero(pred_q-10, axis=0))[0][0]
        # fcol = np.asarray(np.where(pred_q == 10)).T[-1][1]
        # frow, fcol =
        a = np.copy(pred_q[frow:, fcol:])
        a[a == 10] = 0
        # b = ndimage.morphology.grey_closing(a, size=(1,1)).astype(int)
        plt.subplot(2, 2, 4)
        plt.imshow(a,
                   cmap=cmap, norm=norm)
        # plt.subplot(1,2,2)
        # plt.imshow(y_qry[:img_sz0].cpu().numpy().reshape(img_sz, img_sz),
        #            cmap=cmap, norm=norm)
        plt.savefig(f'./model_preds/epoch_30_preds_{im_num}.png')
        # plt.show(block=False)
        # plt.pause(10)
        plt.close()
        # accs = np.array(corrects) / querysz
        accs = np.array(corrects) / (logits.shape[0])

        return accs, pred_q


def generate_weights(meta_params):
    w = {}
    for key in meta_params['mean'].keys():
        eps_sampled = torch.randn(
            meta_params['mean'][key].shape, device=device)
        w[key] = meta_params['mean'][key] + eps_sampled * \
            torch.exp(meta_params['logSigma'][key])

    return w


def initialise_dict_of_dict(key_list):
    q = dict.fromkeys(['mean', 'logSigma'])
    for para in q.keys():
        q[para] = {}
        for key in key_list:
            q[para][key] = 0
    return q
