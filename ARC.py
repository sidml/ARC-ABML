import torch
from torch.utils.data import Dataset
import numpy as np
import json
from glob import glob


class ARC(Dataset):

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, imgsz):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """
        super(ARC, self).__init__()
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        # number of samples per set for evaluation
        self.querysz = self.n_way * self.k_query

        # self.evaluation_path = self.data_path / 'evaluation'
        # self.test_path = self.data_path / 'test'
        self.out_rows, self.out_cols = imgsz, imgsz
        task_paths = f'{root}/training/*.json'
        self.query_x_batch, self.query_y_batch = self.create_batch(
            glob(task_paths), batchsz)

        task_paths = glob(f'{root}/evaluation/*.json')
        test_task_ids = list(map(lambda x: x.split('/')[-1], glob(f'{root}/test/*.json')))
        task_paths = [tp for tp in task_paths if tp.split('/')[-1] not in test_task_ids]
        self.support_x_batch, self.support_y_batch = self.create_batch(
            task_paths, batchsz)
        print('self.support_x_batch', len(self.support_x_batch))
        print('self.query_x_batch', len(self.query_x_batch))

    def pad_im(self, task, out_rows, out_cols, cval=10):

        ip = []
        op = []
        num_pairs = self.k_shot
        for mode in ['train']:
            if len(task[mode]) < num_pairs:
                print('ignoring task, task_len:', len(task), 'required:', num_pairs)
                return 0, 0, 1
            # num_pairs = len(task[mode])
            input_im = np.zeros((num_pairs, 1, out_rows, out_cols))
            output_im = np.zeros(
                (num_pairs, 1, out_rows, out_cols), dtype=np.long)
            for task_num in range(num_pairs):
                im = np.array(task[mode][task_num]['input'])
                nrows, ncols = im.shape
                if (nrows > out_rows) or (ncols > out_cols):
                    return 0, 0, 1
                im = np.pad(im, ((out_rows-nrows, 0), (out_cols-ncols, 0)), mode='constant',
                            constant_values=(cval, cval))

                input_im[task_num, 0] = im
                im = np.array(task[mode][task_num]['output'])
                nrows, ncols = im.shape
                if (nrows > out_rows) or (ncols > out_cols):
                    return 0, 0, 1
                im = np.pad(im, ((out_rows-nrows, 0), (out_cols-ncols, 0)), mode='constant',
                            constant_values=(cval, cval))
                output_im[task_num, 0] = im
            ip.extend(input_im)
            op.extend(output_im)

        return np.vstack(ip), np.vstack(op), 0

    def pad_im_test(self, task, out_rows, out_cols, cval=10):

        ip = []
        for mode in ['train', 'test']:
            num_pairs = len(task[mode])
            input_im = np.zeros((num_pairs, 1, out_rows, out_cols))
            num_pairs = len(task[mode])
            for task_num in range(num_pairs):
                im = np.array(task[mode][task_num]['input'])
                nrows, ncols = im.shape
                im = np.pad(im, ((out_rows-nrows, 0), (out_cols-ncols, 0)), mode='constant',
                            constant_values=(cval, cval))

                input_im[task_num, 0] = im
            ip.extend(input_im)
        return np.vstack(ip)

    def create_batch(self, task_paths, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        x_batch = []  # support set batch
        y_batch = []  # support set batch
        for task_file in task_paths[:batchsz]:
            with open(task_file, 'r') as f:
                task = json.load(f)

            input_im, output_im, not_valid = self.pad_im(task, self.out_rows,
                                                         self.out_cols)
            if not_valid:
                continue
            x_batch.extend(input_im[None])
            y_batch.extend(output_im[None])
        return x_batch, y_batch

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        support_x = torch.tensor(
            self.support_x_batch[index], dtype=torch.float32)
        support_y = torch.tensor(
            self.support_y_batch[index], dtype=torch.long)

        query_x = torch.tensor(self.query_x_batch[index], dtype=torch.float32)
        query_y = torch.tensor(self.query_y_batch[index], dtype=torch.long)
        return support_x[:, None], support_y.reshape(-1),\
            query_x[:, None], query_y.reshape(-1)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        # return self.batchsz
        return min(len(self.query_x_batch), len(self.support_x_batch))


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from matplotlib import pyplot as plt
    from matplotlib import colors

    mini = ARC('../mini-imagenet/', mode='train', n_way=5,
               k_shot=1, k_query=1, batchsz=400)

    print('len(mini)', len(mini))
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF'])
    norm = colors.Normalize(vmin=0, vmax=10)
    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_
        img_sz = support_x.shape[1]*support_x.shape[2]*support_x.shape[3]
        plt.figure()
        for i, x in enumerate(support_x):
            plt.subplot(len(support_x), 2, i*2+1)
            plt.imshow(x[0], cmap=cmap, norm=norm)
            plt.title('support_x')
            plt.subplot(len(support_x), 2, i*2+2)
            y = support_y[i*img_sz:(i+1)*img_sz]
            plt.imshow(
                y.reshape(support_x.shape[2], support_x.shape[3]), cmap=cmap, norm=norm)
            plt.title('support_y')

        plt.tight_layout()
        plt.show()
