import torch
import torch.nn as nn

from mmcv.cnn import build_norm_layer


def select_activation(activation_type):
    if activation_type == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif activation_type == "relu":
        return nn.ReLU(inplace=True)
    elif activation_type == "prelu":
        return nn.PReLU()
    elif activation_type == "none":
        return nn.Identity()
    else:
        print("activation type {} is not supported".format(activation_type))
        raise NotImplementedError


def std_pooling(batch, batch_mean, dim=-1, unbiased=False, eps=1e-8):
    # adding epsilon in sqrt function to make more numerically stable results (yufeng)
    r2 = torch.sum((batch - batch_mean.unsqueeze(-1))**2, dim)
    if unbiased:
        length = batch.shape[dim] - 1
    else:
        length = batch.shape[dim]
    return torch.sqrt(r2/length + eps)


class Stats_pooling(nn.Module):
    def __init__(self, input_dim=1500):
        super(Stats_pooling, self).__init__()
        self.out_dim = 2 * input_dim

    def forward(self, x):
        """
        x.size() = [batch_size, feature_dim, seq_length]
        """
        mean_frame = torch.mean(x, -1, False)
        if self.training:
            std_frame = std_pooling(x, mean_frame, -1, False)
        else:
            std_frame = torch.std(x, -1, False)
        output = torch.cat([mean_frame, std_frame], dim=-1)
        # print(output.shape)
        output = output.view(-1, self.out_dim)
        return output


class StatsPooling(nn.Module):
    """Stats Pooling neck.
    """
    def __init__(self, in_plane, emb_dim, emb_bn=True, emb_affine=True,
                 activation_type="relu", norm_type="BN1d", output_stage=(0,)):
        super(StatsPooling, self).__init__()
        self.avgpool = Stats_pooling(in_plane)
        embedding = []
        initial_dim = self.avgpool.out_dim
        self.output_stage = output_stage
        if isinstance(emb_dim, list):
            self.stages = len(emb_dim)
            for e_dim, do_bn, do_affine, act_type in zip(emb_dim, emb_bn, emb_affine, activation_type):
                fc = [nn.Linear(initial_dim, e_dim)]
                initial_dim = e_dim
                fc.append(select_activation(act_type))
                if do_bn:
                    cfg = dict(type=norm_type, requires_grad=True, momentum=0.5, affine=do_affine)
                    fc.append(build_norm_layer(cfg, e_dim)[1])
                embedding.append(nn.Sequential(*fc))
        else:
            self.stages = 1
            embedding.append(nn.Linear(initial_dim, emb_dim))
            embedding.append(select_activation(activation_type))
            if emb_bn:
                cfg = dict(type=norm_type, requires_grad=True, momentum=0.5, affine=emb_affine)
                embedding.append(build_norm_layer(cfg, emb_dim)[1])
        self.embedding = nn.Sequential(*embedding)

    def init_weights(self):
        pass

    def forward(self, inputs):
        out = self.avgpool(inputs)
        if self.stages > 1 and len(self.output_stage) > 1 and self.training:
            # contains more than one fc layers and needs to output more than one vector and training mode
            embs = []
            for fc in self.embedding:
                out = fc(out)
                embs.append(out)
            results = []
            for stage in self.output_stage:
                results.append(embs[stage])
            return tuple(results)
        else:
            return self.embedding(out)


class StatsPoolingMSEA(nn.Module):
    """Stats Pooling neck.
    """
    def __init__(self, in_plane, emb_dim, emb_bn=True, emb_affine=True,
                 activation_type="relu", norm_type="BN1d", output_stage=(0,)):
        super(StatsPoolingMSEA, self).__init__()
        assert isinstance(in_plane, tuple)
        self.avgpool = [Stats_pooling(plane) for plane in in_plane]
        embedding = []
        initial_dim = sum([pool.out_dim for pool in self.avgpool])
        self.output_stage = output_stage
        if isinstance(emb_dim, list):
            self.stages = len(emb_dim)
            for e_dim, do_bn, do_affine, act_type in zip(emb_dim, emb_bn, emb_affine, activation_type):
                fc = [nn.Linear(initial_dim, e_dim)]
                initial_dim = e_dim
                fc.append(select_activation(act_type))
                if do_bn:
                    cfg = dict(type=norm_type, requires_grad=True, momentum=0.5, affine=do_affine)
                    fc.append(build_norm_layer(cfg, e_dim)[1])
                embedding.append(nn.Sequential(*fc))
        else:
            self.stages = 1
            embedding.append(nn.Linear(initial_dim, emb_dim))
            embedding.append(select_activation(activation_type))
            if emb_bn:
                cfg = dict(type=norm_type, requires_grad=True, momentum=0.5, affine=emb_affine)
                embedding.append(build_norm_layer(cfg, emb_dim)[1])
        self.embedding = nn.Sequential(*embedding)

    def init_weights(self):
        pass

    def forward(self, inputs):
        out = [pool(inp) for pool, inp in zip(self.avgpool, inputs)]
        out = torch.cat(out, dim=-1)
        if self.stages > 1 and len(self.output_stage) > 1 and self.training:
            # contains more than one fc layers and needs to output more than one vector and training mode
            embs = []
            for fc in self.embedding:
                out = fc(out)
                embs.append(out)
            results = []
            for stage in self.output_stage:
                results.append(embs[stage])
            return tuple(results)
        else:
            return self.embedding(out)