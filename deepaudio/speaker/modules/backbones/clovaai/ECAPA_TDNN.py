import math
import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=4):

        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes / scale))

        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)

        self.nums = scale - 1

        convs = []
        bns = []

        num_pad = math.floor(kernel_size / 2) * dilation

        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU()

        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)

        out += residual

        return out


class Res2Net(nn.Module):

    def __init__(self, block, C, model_scale, nOut, n_mels, encoder_type, context=False, summed=False,
                 out_bn=False, **kwargs):

        self.context = context
        self.summed = summed
        self.n_mfcc = n_mels
        self.encoder_type = encoder_type
        self.out_bn = out_bn

        super(Res2Net, self).__init__()
        self.scale = model_scale

        self.conv1 = nn.Conv1d(self.n_mfcc, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)

        self.layer1 = block(C, C, kernel_size=3, dilation=2, scale=self.scale)
        self.layer2 = block(C, C, kernel_size=3, dilation=3, scale=self.scale)
        self.layer3 = block(C, C, kernel_size=3, dilation=4, scale=self.scale)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        if self.context:
            attn_input = 1536 * 3
        else:
            attn_input = 1536

        if self.encoder_type == 'ECA':
            attn_output = 1536
        elif self.encoder_type == 'ASP':
            attn_output = 1
        else:
            raise ValueError('Undefined encoder')

        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.bn5 = nn.BatchNorm1d(3072)

        self.fc6 = nn.Linear(3072, nOut)
        self.bn6 = nn.BatchNorm1d(nOut)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        if self.summed:
            x1 = self.layer1(x)
            x2 = self.layer2(x + x1)
            x3 = self.layer3(x + x1 + x2)
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        if self.context:
            global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                                  torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        else:
            global_x = x

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)

        x = self.bn5(x)

        x = self.fc6(x)

        if self.out_bn:
            x = self.bn6(x)

        return x


def MainModel(configs):
    model = Res2Net(Bottle2neck,
                    C=configs.model.channels,
                    n_mels=configs.feature.n_mel,
                    model_scale=configs.model.model_scale,
                    nOut=configs.model.embed_dim,
                    context=configs.model.context,
                    summed=configs.model.summed,
                    out_bn=configs.model.out_bn,
                    encoder_type=configs.model.encoder_type)
    return model


if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim
    def MainModel(**kwargs):
        model = Res2Net(Bottle2neck, C=1024, n_mels=80, model_scale=8, nOut=192, context=True, summed=True,
                        encoder_type='ECA', **kwargs)
        return model


    x = torch.zeros(2, 200, 80)
    model = MainModel()
    out = model(x)
    print(out.shape)  # should be [2, 192]
