import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel_1
from model.models.INSTA import INSTA

"""
The INSTA_ProtoNet class combines INSTA-based attention mechanisms with the prototypical networks approach.
This hybrid model is designed for few-shot learning tasks where it's important to quickly adapt to new classes
with very few examples per class.
"""

class INSTA_ProtoNet(FewShotModel_1):
    def __init__(self, args):
        """
        Initializes the INSTA_ProtoNet with the given arguments.
        
        Parameters:
        - args: Configuration settings including hyperparameters for the network setup.
        """
        super().__init__(args)
        self.args = args
        # Instantiate the INSTA model with specific parameters.
        self.INSTA = INSTA(640, 5, 0.2, 3, args=args)

    def inner_loop(self, proto, support):
        """
        Performs an inner optimization loop to fine-tune prototypes on support sets during meta-training.
        
        Parameters:
        - proto: Initial prototypes, typically the mean of the support embeddings.
        - support: Support set embeddings used for fine-tuning the prototypes.
        
        Returns:
        - SFC: Updated (fine-tuned) prototypes.
        """
        # Clone and detach prototypes to prevent gradients from accumulating across episodes.
        SFC = proto.clone().detach()
        SFC = nn.Parameter(SFC, requires_grad=True)

        # Initialize an SGD optimizer specifically for this inner loop.
        optimizer = torch.optim.SGD([SFC], lr=0.6, momentum=0.9, dampening=0.9, weight_decay=0)

        # Create labels for the support set, used in cross-entropy loss during fine-tuning.
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label_shot = label_shot.type(torch.cuda.LongTensor)
        
        # Perform gradient steps to update the prototypes.
        with torch.enable_grad():
            for k in range(50):  # Number of gradient steps.
                rand_id = torch.randperm(self.args.way * self.args.shot).cuda()
                for j in range(0, self.args.way * self.args.shot, 4):
                    selected_id = rand_id[j: min(j + 4, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.classifier(batch_shot.detach(), SFC)
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    def classifier(self, query, proto):
        """
        Simple classifier that computes the negative squared Euclidean distance between query and prototype vectors,
        scaled by a temperature parameter for controlling the sharpness of the distribution.
        
        Parameters:
        - query: Query set embeddings.
        - proto: Prototype vectors.

        Returns:
        - logits: Logits representing similarity scores between each query and each prototype.
        """
        logits = -torch.sum((proto.unsqueeze(0) - query.unsqueeze(1)) ** 2, 2) / self.args.temperature
        return logits.squeeze()

    def _forward(self, instance_embs, support_idx, query_idx):
        """
        Forward pass of the model, processing both support and query data.
        
        Parameters:
        - instance_embs: Embeddings of all instances.
        - support_idx: Indices identifying support instances.
        - query_idx: Indices identifying query instances.
        
        Implements the forward pass, integrating both spatial and feature adaptation using the INSTA module.
        """
        emb_dim = instance_embs.size()[-3:]
        channel_dim = emb_dim[0]

        # Organize support and query data based on indices, and reshape accordingly.
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + emb_dim))
        query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + emb_dim))
        num_samples = support.shape[1]
        num_proto = support.shape[2]
        support = support.squeeze()

        # Adapt support features using the INSTA model and average to form adapted prototypes.
        adapted_s, task_kernel = self.INSTA(support.view(-1, *emb_dim))
        query = query.view(-1, *emb_dim)
        adapted_proto = adapted_s.view(num_samples, -1, *adapted_s.shape[1:]).mean(0)
        adapted_proto = nn.AdaptiveAvgPool2d(1)(adapted_proto).squeeze(-1).squeeze(-1)

        # Adapt query features using the INSTA unfolding and kernel multiplication approach.
        query_ = nn.AdaptiveAvgPool2d(1)((self.INSTA.unfold(query, int((task_kernel.shape[-1]+1)/2-1), task_kernel.shape[-1]) * task_kernel)).squeeze()
        query = query + query_
        adapted_q = nn.AdaptiveAvgPool2d(1)(query).squeeze(-1).squeeze(-1)

        # Optionally perform an inner loop optimization during testing.
        if self.args.testing:
            adapted_proto = self.inner_loop(adapted_proto, nn.AdaptiveAvgPool2d(1)(support).squeeze().view(num_proto*num_samples, channel_dim))
        
        # Classify using the adapted prototypes and query embeddings.
        logits = self.classifier(adapted_q, adapted_proto)

        if self.training:
            reg_logits = None
            return logits, reg_logits
        else:
            return logits
