import torch.nn as nn
import torch
from models.vit import vit_b_16 as vit_b_16, \
                       vit_t_16 as vit_t_16, \
                       vit_s_16 as vit_s_16
from models.ResNet import resnet50_relu, resnet34_relu
import numpy as np

BASIC_MODEL = {
               "resnet50_relu" : resnet50_relu,
               "resnet34_relu" : resnet34_relu,
               "vit_b_16" : vit_b_16,
               "vit_s_16" : vit_s_16,
               "vit_t_16" : vit_t_16,
                }

class MCPNet_pp(nn.Module):
    def __init__(self, num_classes, basic_model, concept_per_layer, concept_cha, drop_p = 0.5, **basic_kwargs):
        super(MCPNet_pp, self).__init__()
        self.concept_per_layer = nn.Parameter(torch.tensor(concept_per_layer), requires_grad = False)
        self.concept_cha = concept_cha
        self.basic_model = basic_model
        if "vit" not in basic_model:
            basic_kwargs.pop("sel_layers")
        self.feature_extractor = BASIC_MODEL[basic_model.lower()](num_classes = num_classes, **basic_kwargs)
        
        self.pool = nn.AdaptiveMaxPool1d(output_size = 1)
        if "vit" in basic_model:
            assert len(self.concept_per_layer) == len(self.feature_extractor.sel_layers), f"Select layer and concept setting not match !! {self.concept_per_layer} - {self.feature_extractor.sel_layers}"

        if "vit" not in basic_model:
            self.fc = nn.Linear(sum(self.concept_per_layer), num_classes, bias = False)
        else:
            self.fc_patch = nn.Linear(self.feature_extractor.hidden_dim * len(self.concept_per_layer), num_classes)
            self.fc_patch_MCP = nn.Linear(sum(self.concept_per_layer), num_classes, bias = False)
            if isinstance(self.fc_patch, nn.Linear):
                nn.init.zeros_(self.fc_patch.weight)
                nn.init.zeros_(self.fc_patch.bias)

            if isinstance(self.fc_patch_MCP, nn.Linear):
                nn.init.zeros_(self.fc_patch_MCP.weight)

            
        self.concept_idx = torch.cumsum(self.concept_per_layer, dim = 0)
        self.concept_idx = torch.cat([torch.tensor([0]), self.concept_idx], dim = 0)
        self.drop_p = drop_p

    def dropout_l(self, MCP_feat):
        B, L = MCP_feat.shape
        randn_layer = torch.randint(0, len(self.concept_idx) - 1, (MCP_feat.shape[0],)).to(MCP_feat.device)
        masks = torch.ones_like(MCP_feat).to(MCP_feat.device)
        drop_prob = torch.rand(B).to(MCP_feat.device)
        for layer_i, (concept_si, concept_ei) in enumerate(zip(self.concept_idx[:-1], self.concept_idx[1:])):
            masks[:, concept_si : concept_ei] = masks[:, concept_si : concept_ei] * torch.logical_or((randn_layer != layer_i).unsqueeze(1), (drop_prob > self.drop_p).unsqueeze(1))
        MCP_feat = MCP_feat * masks
        # for layer_i, (concept_si, concept_ei) in enumerate(zip(self.concept_idx[:-1], self.concept_idx[1:])):
        return MCP_feat * 1 / (1 - ((self.concept_per_layer[randn_layer] * (drop_prob <= self.drop_p)).unsqueeze(1) / torch.sum(self.concept_per_layer)))
            
    def forward(self, x, concept_vecs = None, concept_means = None):
        feats = self.feature_extractor(x)
        if concept_means is None:
            if "vit" in self.basic_model:
                return feats, None, None, None
            else:
                return feats, None, None
        
        if "vit" in self.basic_model:
            # simply average the patch embedding
            feat_patches = []
            # using the max concept response from the patch embedding
            responses_patch = []
            for layer_i, feat in enumerate(feats):
                feat_patch = feat[:, 1:]
                feat_patches.append(nn.functional.adaptive_avg_pool1d(feat_patch.permute(0, 2, 1), output_size = 1)[..., 0])

                B, N, C = feat_patch.shape
                feat_patch = feat_patch.permute(0, 2, 1).reshape(B, self.concept_per_layer[layer_i], self.concept_cha[layer_i], N)
                feat_patch = feat_patch - concept_means[layer_i].unsqueeze(0).unsqueeze(3)
                response = torch.sum(feat_patch * concept_vecs[layer_i].unsqueeze(0).unsqueeze(3), dim = 2)
                response_pool = self.pool(response)
                responses_patch.append(response_pool[..., 0])
            logits_patch = torch.cat(feat_patches, dim = 1)   
            MCP_feat_patch = torch.cat(responses_patch, dim = 1)
            if self.training:
                logits_patch = self.fc_patch(logits_patch)
                logits_patch_MCP = self.fc_patch_MCP(self.dropout_l(MCP_feat_patch))
            else:
                logits_patch = self.fc_patch(logits_patch)
                logits_patch_MCP = self.fc_patch_MCP(MCP_feat_patch)
            return feats, logits_patch, logits_patch_MCP, MCP_feat_patch
        else:
            responses = []
            response_maps = []
            for layer_i, feat in enumerate(feats):
                feat = feat.flatten(2)
                B, C, N = feat.shape
                feat = feat.reshape(B, self.concept_per_layer[layer_i], self.concept_cha[layer_i], N)
                feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3)
                response = torch.sum(feat * concept_vecs[layer_i].unsqueeze(0).unsqueeze(3), dim = 2)
                response_maps.append(response)
                response_pool = self.pool(response)
                responses.append(response_pool[..., 0])
            MCP_feat = torch.cat(responses, dim = 1)
            
            if self.training:
                logits = self.fc(self.dropout_l(MCP_feat))
            else:
                logits = self.fc(MCP_feat)
            return feats, logits, MCP_feat
    
    def load_pretrained(self, path):
        self.feature_extractor.load_pretrained(path)

def load_weight(path, model):
    parameters = torch.load(path)["Model"]
    model.load_state_dict(parameters)
    
    return model

def load_model(num_classes, basic_model, concept_per_layer, concept_cha, **model_kwargs):
    model = MCPNet_pp(num_classes, basic_model, concept_per_layer, concept_cha, **model_kwargs)
    return model

if __name__ == "__main__":
    model = MCPNet_pp(100, "vit_b_16", [12, 12, 12, 12], [64, 64, 64, 64], "max", sel_layers = [1, 2, 3, 4], use_CLS = False, wo_norm = False, drop_p = 1.0).cuda()
    a = torch.randn([16, 3, 224, 224]).cuda()
    concept_vecs = [torch.randn([12, 64]).cuda(), torch.randn([12, 64]).cuda(), torch.randn([12, 64]).cuda(), torch.randn([12, 64]).cuda()]
    concept_means = [torch.randn([12, 64]).cuda(), torch.randn([12, 64]).cuda(), torch.randn([12, 64]).cuda(), torch.randn([12, 64]).cuda()]
    feats, logits, MCP_f = model(a, concept_vecs, concept_means)