import torch
import torch.nn as nn

class CCC_with_drop_p(nn.Module):
    def __init__(self, concept_per_layer, drop_p = 0.5):
        super(CCC_with_drop_p, self).__init__()
        self.concept_per_layer = concept_per_layer
        self.concept_idx = torch.cumsum(torch.tensor(self.concept_per_layer), dim = 0)
        self.concept_idx = torch.cat([torch.tensor([0]), self.concept_idx], dim = 0)
        self.drop_p = drop_p

    def compute_cross_entropy(self, q, p):
        q = torch.nn.functional.log_softmax(q, dim=-1)
        loss = torch.sum(p * q, dim=-1)
        return - loss.mean()

    def cal_sim(self, x):
        l2_dist = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), p = 2, dim = -1)
        return torch.log((l2_dist + 1) / (l2_dist + 1e-5))# / torch.log(torch.tensor([1 / 1e-5]).cuda())

    def forward(self, MCP_feats, label):
        total_loss = 0
        B = label.shape[0]
        label_mask = (label.unsqueeze(1) == label).float()
        label_mask = torch.diagonal_scatter(label_mask, torch.zeros(label_mask.shape[0]).to(MCP_feats.get_device()), 0)
        label_mask = label_mask / torch.sum(label_mask, dim = 1, keepdim = True).clamp(min = 1)
        MCP_feat_mask = torch.eye(label_mask.shape[0]).to(MCP_feats.get_device())
        randn_layer = torch.randint(0, len(self.concept_idx) - 1, (1,)).to(MCP_feats.device)
        for layer_i, (concept_si, concept_ei) in enumerate(zip(self.concept_idx[:-1], self.concept_idx[1:])):
            if layer_i == randn_layer and torch.rand(1) <= self.drop_p:
                continue
            MCP_feat = MCP_feats[:, concept_si : concept_ei]
            MCP_feat_sim = self.cal_sim(MCP_feat)
            MCP_feat_sim = MCP_feat_sim * (1 - MCP_feat_mask) - MCP_feat_mask * 1e-9
            total_loss = total_loss + self.compute_cross_entropy(MCP_feat_sim, label_mask)
        return total_loss / len(self.concept_per_layer)
