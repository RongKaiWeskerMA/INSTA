import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel_1

"""
The ProtoNet class inherits from FewShotModel_1, which is assumed to be tailored for few-shot learning scenarios.
This implementation specifically targets the scenario where the model is expected to learn from a limited number of examples (support set)
and generalize well to new, unseen examples (query set).
"""

class ProtoNet(FewShotModel_1):
    def __init__(self, args):
        """
        Initialize the ProtoNet with the given arguments.
        This constructor passes any arguments to the superclass FewShotModel_1, which might perform some initial setup.
        """
        super().__init__(args)

    def _forward(self, instance_embs, support_idx, query_idx):
        """
        Custom forward logic for processing instance embeddings and calculating the prototypes.

        Parameters:
        - instance_embs: Tensor containing embeddings for all instances.
        - support_idx: Indices of support examples within instance_embs.
        - query_idx: Indices of query examples within instance_embs.

        The method handles two cases:
        1. If Grad-CAM is enabled, it returns the raw embeddings for visualization purposes.
        2. Otherwise, it processes the embeddings to compute class prototypes and their distances to query examples.
        """
        if self.args.grad_cam:
            # Return embeddings directly for Grad-CAM visualization.
            return instance_embs

        else:
            # Extract the size of the last dimension, which represents the dimensionality of the embeddings.
            emb_dim = instance_embs.size(-1)

            # Organize support and query data by reshaping them according to their indices.
            support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
            query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))

            # Compute the mean of the support embeddings to form the prototypes for each class.
            proto = support.mean(dim=1)  # Ntask x NK x d

            # Prepare for distance calculation between queries and prototypes.
            num_batch = proto.shape[0]
            num_proto = proto.shape[1]
            num_query = np.prod(query_idx.shape[-2:])

            if True:  # Placeholder for a boolean flag such as self.args.use_euclidean
                # Compute Euclidean distances
                query = query.view(-1, emb_dim).unsqueeze(1)  # Reshape for broadcasting
                proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
                proto = proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)
                logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
            else:
                # Compute Cosine similarity
                proto = F.normalize(proto, dim=-1)  # Normalize for cosine distance
                query = query.view(num_batch, -1, emb_dim)  # Reshape for matrix multiplication
                logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.args.temperature
                logits = logits.view(-1, num_proto)

            # Depending on the training state, return logits directly or with additional processing.
            if self.training:
                return logits, None
            else:
                return logits
