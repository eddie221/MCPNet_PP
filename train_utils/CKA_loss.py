import torch.nn as nn
import torch

class CKA_loss_sampled(nn.Module):
    def __init__(self, concept_per_layer):
        super(CKA_loss_sampled, self).__init__()
        self.concept_per_layer = concept_per_layer  
    
    def __repr__(self):
        basic = super().__repr__()
        str_show = f"{basic[:-1]}concept_per_layer={self.concept_per_layer})"
        return str_show

    def unbiased_HSIC(self, x, y):
        #create the unit **vector** filled with ones
        n = x.shape[1]
        ones = torch.ones(x.shape[0], n, 1).cuda()

        #fill the diagonal entries with zeros 
        mask = torch.eye(n).unsqueeze(0).cuda()
        x = x * (1 - mask)
        y = y * (1 - mask)

        #first part in the square brackets
        trace = torch.sum(torch.matmul(x, y.permute(0, 2, 1)) * mask, dim = (-1, -2), keepdim = True)

        #middle part in the square brackets
        nominator1 = torch.sum(x, dim = (-2, -1), keepdim = True)
        nominator2 = torch.sum(y, dim = (-2, -1), keepdim = True)
        denominator = (n - 1) * (n - 2)
        middle = torch.matmul(nominator1, nominator2) / denominator
        
        #third part in the square brackets
        multiplier1 = 2 / (n - 2)
        multiplier2 = torch.matmul(torch.matmul(ones.permute(0, 2, 1), x), torch.matmul(y, ones))
        last = multiplier1 * multiplier2

        #complete equation
        unbiased_hsic = 1 / (n * (n - 3)) * (trace + middle - last)
        return unbiased_hsic
    
    def CKA(self, kernel):
        # random select n case to minimize (reducing the memory usage)
        index = torch.triu_indices(kernel.shape[0], kernel.shape[0], 1).to(kernel.get_device())
        index = index.permute(1, 0)
        rand_idx = torch.randperm(index.shape[0])[:kernel.shape[0]].to(kernel.get_device())
        torch.distributed.broadcast(rand_idx, src = 0)
        index = index[rand_idx].permute(1, 0)
        nominator = self.unbiased_HSIC(kernel[index[0]], kernel[index[1]])
        denominator1 = self.unbiased_HSIC(kernel[index[0]], kernel[index[0]])
        denominator2 = self.unbiased_HSIC(kernel[index[1]], kernel[index[1]])
        denominator1 = torch.nn.functional.relu(denominator1)
        denominator2 = torch.nn.functional.relu(denominator2)
        denominator = denominator1 * denominator2
        mask = (denominator != 0)
        cka = (nominator * mask) / torch.sqrt(torch.clamp(denominator, min = 1e-8))
        return cka
    
    def forward(self, concept_pools):
        # calculate the concept number and channel number of each concept
        CKA_loss = 0
        for layer_i, concept_blocks in enumerate(concept_pools):
            # concept_blocks = concept_pool[:, :cha_per_concept[step] * concept_num[step]]
            if len(concept_blocks.shape) == 4: 
                B, C, H, W = concept_blocks.shape
                concept_blocks = torch.flatten(concept_blocks.reshape(B, self.concept_per_layer[layer_i], -1, H, W).permute(1, 0, 2, 3, 4), 2)
            elif len(concept_blocks.shape) == 3: 
                # N => patch number
                B, C, N = concept_blocks.shape
                concept_blocks = torch.flatten(concept_blocks.reshape(B, self.concept_per_layer[layer_i], -1, N).permute(1, 0, 2, 3), 2)
            concept_blocks_kernel = torch.matmul(concept_blocks, concept_blocks.permute(0, 2, 1))
            CKA_loss = CKA_loss + torch.mean(torch.abs(self.CKA(concept_blocks_kernel)))
        return CKA_loss

    