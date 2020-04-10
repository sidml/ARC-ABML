import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, feat_dim=1, out_rows=30, out_cols=30, n_classes=10,
                 dropout_rate=0.25):
        super(CNN, self).__init__()

        self.num_classes = n_classes
        self.out_cols = out_cols
        self.out_rows = out_rows
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.Conv2d(feat_dim, 64, kernel_size=5, padding=2),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=dropout_rate),
        )

        self.final_layer = nn.Linear(16, out_rows*out_cols*n_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:]).reshape(x.shape[0], -1)
        return self.final_layer(x).reshape(-1, self.num_classes)

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()
