import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel_1
from model.models.INSTA import INSTA
# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class INSTA_ProtoNet(FewShotModel_1):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.INSTA = INSTA(640, 5, 0.2, 3, args=args)

    def inner_loop(self, proto, support):
        # init the proto
        SFC = proto.clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)
        # the learning rate which yields the best resulat is 0.6 yet.
        optimizer = torch.optim.SGD([SFC], lr=0.006, momentum=0.9, dampening=0.9, weight_decay=0)
        # crate label for finetune
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label_shot = label_shot.type(torch.cuda.LongTensor)
        with torch.enable_grad():
            for k in range(0, 50):
                rand_id = torch.randperm(self.args.way * self.args.shot).cuda()
                for j in range(0, self.args.way * self.args.shot, 4):
                    selected_id = rand_id[j: min(j + 4, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.classifier(batch_shot.detach(), SFC)
                    if logits.dim()==1: logits = logits.unsqueeze(0)
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    def classifier(self, query, proto):
        logits = - torch.sum((proto.unsqueeze(0) - query.unsqueeze(1)) ** 2, 2)/self.args.temperature
        return logits.squeeze()

    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size()[-3:]
        channel_dim = emb_dim[0]

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + emb_dim))
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + emb_dim))
        num_samples = support.shape[1]
        num_proto = support.shape[2]
        support = support.squeeze()

        # adapted_support and task-specific kernels
        adapted_s, task_kernel = self.INSTA(support.view(-1, *emb_dim))
        query = query.view(-1, *emb_dim)
        adapted_proto = adapted_s.view(num_samples, -1, *adapted_s.shape[1:]).mean(0)
        adapted_proto = nn.AdaptiveAvgPool2d(1)(adapted_proto).squeeze(-1).squeeze(-1)

        ## The query feature map adaptation
        query_ = nn.AdaptiveAvgPool2d(1)((self.INSTA.unfold(query, int((task_kernel.shape[-1]+1)/2-1), task_kernel.shape[-1]) * task_kernel)).squeeze()
        query = query + query_
        adapted_q = nn.AdaptiveAvgPool2d(1)(query).squeeze(-1).squeeze(-1)  ##put squeeze if we dont use the feature map in the end
        if self.args.testing:
            adapted_proto = self.inner_loop(adapted_proto, nn.AdaptiveAvgPool2d(1)(support).squeeze().view(num_proto*num_samples,channel_dim))
        logits = self.classifier(adapted_q, adapted_proto)

        if self.training:
            reg_logits = None
            return logits, reg_logits
        else:
            return logits
