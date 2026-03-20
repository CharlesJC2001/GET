import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from scipy.optimize import linear_sum_assignment
from typing import Optional


@torch.jit.script
def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob)**gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum('nc,mc->nm', focal_pos, targets) + torch.einsum('nc,mc->nm', focal_neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss


def get_iou(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


@torch.jit.script
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


@torch.jit.script
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss.mean()


@torch.jit.script
def dice_loss_multi_calsses(input: torch.Tensor,
                            target: torch.Tensor,
                            epsilon: float = 1e-5,
                            weight: Optional[float] = None) -> torch.Tensor:
    r"""
    modify compute_per_channel_dice from
    https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # convert the feature channel(category channel) as first
    input = input.permute(1, 0)
    target = target.permute(1, 0)

    target = target.float()
    # Compute per channel Dice Coefficient
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / (
        torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)

    loss = 1.0 - per_channel_dice

    return loss.mean()


# def contrast_loss(input: torch.Tensor): # input(num_inst, media)
#     # infoNCE loss + MSEloss
#     temperature = 0.1
#     num_inst = input.size(0)
#     features = F.normalize(input, dim=1, p=2.0)
#     similarity_matrix = torch.matmul(features, features.T) # (num_inst, num_inst)
    
#     mseloss = nn.MSELoss(reduction='mean')
#     target = torch.eye(num_inst)
#     mse_loss = mseloss(similarity_matrix, target)

#     similarity_matrix_modified = similarity_matrix / temperature
#     softmax_matrix = F.softmax(similarity_matrix_modified, dim=1)
#     diagonal = torch.diagonal(softmax_matrix)
#     log_diagonal = torch.log(diagonal+1e-3)
#     contrast_loss = (-1) * log_diagonal.sum() / num_inst
#     loss = contrast_loss + mse_loss
#     loss = mse_loss

#     return loss


# @torch.jit.script
# def contrast_loss(input: torch.Tensor): # input(num_inst, media)
#     # infoNCE loss
#     temperature = 0.05
#     num_inst = input.size(0)
#     features = F.normalize(input, dim=1, p=2.0)
#     similarity_matrix = torch.matmul(features, features.T) # (num_inst, num_inst)
#     similarity_matrix_modified = similarity_matrix / temperature
#     softmax_matrix = F.softmax(similarity_matrix_modified, dim=1)
#     diagonal = torch.diagonal(softmax_matrix)
#     log_diagonal = torch.log(diagonal+1e-3)
#     loss = (-1) * log_diagonal.sum() / num_inst

#     return loss

# def contrast_loss(input: torch.Tensor): # input(num_inst, media)
#     # infoNCE loss
#     device = input.device
#     temperature = 0.1
#     num_inst = input.size(0)
#     input = F.normalize(input, dim=1, p=2.0)
#     similarity_matrix = torch.matmul(input, input.T) # (num_inst, num_inst)
#     similarity_matrix_modified = similarity_matrix / temperature
#     crossentropy = nn.CrossEntropyLoss()
#     target = torch.eye(num_inst, device=device)
#     infoNCE_loss = crossentropy(similarity_matrix_modified, target)
#     loss = infoNCE_loss

#     return loss

def contrast_loss(input: torch.Tensor, target: torch.Tensor): # input(M, media), target(M,M)
    # MSE loss

    input = F.normalize(input, dim=1, p=2.0)
    similarity_matrix = torch.matmul(input, input.T) # (M, M)
    MSE_loss = F.mse_loss(similarity_matrix, target, reduction='mean')
    loss = MSE_loss

    return loss

# @torch.jit.script
# def contrast_loss(input: torch.Tensor): # input(num_inst, media)
#     # bce loss
#     num_inst = input.size(0)
#     device = input.device
#     # input = F.normalize(input, dim=1, p=2.0)
#     similarity_matrix = torch.matmul(input, input.T) # (num_inst, num_inst)
#     target = torch.eye(num_inst, device=device)
#     bce_loss = F.binary_cross_entropy_with_logits(similarity_matrix, target)

#     loss = bce_loss

#     return loss


# def contrast_loss(input: torch.Tensor): # input(num_inst, media)
#     # MSEloss
#     num_inst = input.size(0)
#     features = F.normalize(input, dim=1, p=2.0)
#     similarity_matrix = torch.matmul(features, features.T) # (num_inst, num_inst)
#     device = input.device
    
#     mseloss = nn.MSELoss(reduction='mean')
#     target = torch.eye(num_inst, device=device)
#     mse_loss = mseloss(similarity_matrix, target)
#     loss = mse_loss

#     return loss

# def log_barrier(input: torch.Tensor): # input(num_inst, media)
#     num_inst = input.size(0)
#     reg_weight = 0.01
#     epsilon = 1e-8
#     log_barrier_reg = -reg_weight * torch.sum(torch.log(input**2+epsilon)) / num_inst
#     loss = log_barrier_reg

#     return loss

def CoSENT_loss(feats: torch.Tensor, target: torch.Tensor): # feats(M, media), inst_mask(M, M)

    # calculate similarity_matrix
    lambda_ = 20
    feats_norm = F.normalize(feats, dim=1, p=2.0)
    similarity_matrix = torch.matmul(feats_norm, feats_norm.T) # (M, M)
    # calculate CoSENT_loss
    target_1 = target.clone()
    target_1.fill_diagonal_(0) 
    similarity_1 = similarity_matrix * target_1
    target_0 = 1-target
    target_0_bool = target_0.to(torch.bool)
    pos_average = torch.sum(similarity_1, dim=1, keepdim=True) / (torch.sum(target_1, dim=1, keepdim=True)+1e-8) # (M, )
    pos_average[pos_average==0] = 0.8 # if a instance only contains one superpoint, we hope it to be a noise
    neg = torch.exp((similarity_matrix-pos_average) * lambda_)
    neg_all = torch.sum(neg[target_0_bool]) / torch.sum(target_0)
    loss = torch.log(1 + neg_all)
    return loss

def CoSENT_3std(feats: torch.Tensor, target: torch.Tensor): # feats(M, media), inst_mask(M, M)

    # calculate similarity_matrix
    lambda_ = 20
    pos_ave = []
    pos_std = []
    feats_norm = F.normalize(feats, dim=1, p=2.0)
    similarity_matrix = torch.matmul(feats_norm, feats_norm.T) # (M, M)
    # calculate CoSENT_loss(3std)
    target_1 = target.clone()
    target_1.fill_diagonal_(0)
    target_1_bool = target_1.to(torch.bool)
    target_0 = 1-target
    target_0_bool = target_0.to(torch.bool)
    for i in range(similarity_matrix.size(0)):
        pos = similarity_matrix[i][target_1_bool[i]]
        mean = torch.mean(pos)
        std = torch.std(pos, unbiased=False)
        mean = torch.nan_to_num(mean, nan=0.3)
        std = torch.nan_to_num(std, nan=0.0)
        pos_ave.append(mean)
        pos_std.append(std)
    pos_ave = torch.tensor(pos_ave).unsqueeze(-1) # (num_inst, 1)
    pos_std = torch.tensor(pos_std).unsqueeze(-1) # (num_inst, 1)
    neg = torch.exp((similarity_matrix-pos_ave+3*pos_std) * lambda_)
    neg_all = torch.sum(neg[target_0_bool]) / torch.sum(target_0)
    loss = torch.log(1 + neg_all)
    return loss

def CoSENT_Kpos(feats: torch.Tensor, target: torch.Tensor): # feats(M, media), tgt_mask(num_inst, M)
    
    lambda_pos = 20
    lambda_neg = 20
    pos_ave = []
    pos_std = []
    num_per_inst = torch.sum(target, dim=1).unsqueeze(-1) # (num_inst, 1)
    feats_expanded = feats.unsqueeze(0) # (1, M, media)
    target_expanded = target.unsqueeze(-1) # (num_inst, M, 1)
    inst_feats_expanded = feats_expanded * target_expanded # (num_inst, M, media)
    inst_feats = inst_feats_expanded.sum(dim=1) / num_per_inst # (num_inst, media)
    # calculate similarity_matrix
    inst_feats_norm = F.normalize(inst_feats, dim=1, p=2.0)
    feats_norm = F.normalize(feats, dim=1, p=2.0)
    similarity_matrix = torch.matmul(inst_feats_norm, feats_norm.T) # (num_inst, M)
    # calculate CoSENT_Kpos
    target_1 = target.clone()
    target_1_bool = target_1.to(torch.bool)
    target_0 = 1-target
    target_0_bool = target_0.to(torch.bool)
    for i in range(similarity_matrix.size(0)):
        pos = similarity_matrix[i][target_1_bool[i]]
        mean = torch.mean(pos)
        std = torch.std(pos, unbiased=False)
        mean = torch.nan_to_num(mean, nan=0.3)
        std = torch.nan_to_num(std, nan=0.0)
        pos_ave.append(mean)
        pos_std.append(std)
    pos_ave = torch.tensor(pos_ave).unsqueeze(-1) # (num_inst, 1)
    pos_std = torch.tensor(pos_std).unsqueeze(-1) # (num_inst, 1)
    pos_exp = torch.exp((similarity_matrix-3*pos_std) * lambda_pos)
    pos_all = torch.sum(pos_exp[target_1_bool]) / torch.sum(target_1)
    neg_exp = torch.exp((3*pos_std-similarity_matrix) * lambda_neg)
    neg_all = torch.sum(neg_exp[target_0_bool]) / torch.sum(target_0)
    loss = torch.log(1 + pos_all + neg_all)

    return loss

def CoSENT_loss_remark(feats: torch.Tensor, target: torch.Tensor): # feats(M, media), inst_mask(M, M)

    # calculate similarity_matrix
    lambda_ = 20
    margin = torch.tensor(0.2)
    feats_norm = F.normalize(feats, dim=1, p=2.0)
    similarity_matrix = torch.matmul(feats_norm, feats_norm.T) # (M, M)
    # calculate CoSENT_loss
    target_1 = target.clone()
    target_1.fill_diagonal_(0) 
    similarity_1 = similarity_matrix * target_1
    target_0 = 1-target
    target_0[torch.where(target_1.sum(-1)==0)]=0
    target_0_bool = target_0.to(torch.bool)
    pos_average = torch.sum(similarity_1, dim=1, keepdim=True) / (torch.sum(target_1, dim=1, keepdim=True)+1e-8) # (M, )
    pos_average[pos_average==0] = 0.8 # if a instance only contains one superpoint OR belongs to the background, we hope it to be a noise
    neg = torch.exp((similarity_matrix-pos_average+margin) * lambda_)
    # neg = torch.exp((similarity_matrix-pos_average) * lambda_)
    neg_all = torch.sum(neg[target_0_bool]) / torch.sum(target_0)
    loss = torch.log(1 + neg_all)
    return loss

# def CoSENT_loss_remark_kpos(feats: torch.Tensor, target: torch.Tensor): # feats(M, media), inst_mask(M, M)

#     # calculate similarity_matrix
#     lambda_neg = 20
#     lambda_pos = 20
#     # margin_neg = torch.tensor(0.2)
#     margin_pos = torch.tensor(0.8)
#     # feats_norm = F.normalize(feats, dim=1, p=2.0)
#     # similarity_matrix = torch.matmul(feats_norm, feats_norm.T) # (M, M)
#     similarity_matrix = torch.matmul(feats, feats.T).sigmoid() # (M, M)
#     # calculate CoSENT_loss
#     target_1 = target.clone()
#     target_1.fill_diagonal_(0)
#     target_1_bool = target_1.to(torch.bool)
#     similarity_1 = similarity_matrix * target_1
#     target_0 = 1-target
#     target_0[torch.where(target_1.sum(-1)==0)]=0
#     target_0_bool = target_0.to(torch.bool)
#     pos_average = torch.sum(similarity_1, dim=1, keepdim=True) / (torch.sum(target_1, dim=1, keepdim=True)+1e-8) # (M, )
#     pos_average[pos_average==0] = 0.8 # if a instance only contains one superpoint OR belongs to the background, we hope it to be a noise
#     # neg = torch.exp((similarity_matrix-pos_average+margin_neg) * lambda_neg)
#     neg = torch.exp((similarity_matrix-pos_average) * lambda_neg)
#     neg_all = torch.sum(neg[target_0_bool]) / torch.sum(target_0)
#     pos = torch.exp((margin_pos-similarity_matrix) * lambda_pos)
#     pos_all = torch.sum(pos[target_1_bool]) / torch.sum(target_1)
#     loss = torch.log(1 + neg_all + pos_all)
#     return loss

def CoSENT_loss_remark_kpos(feats: torch.Tensor, target: torch.Tensor): # feats(M, media), inst_mask(M, M)

    # calculate similarity_matrix
    lambda_neg = 20
    lambda_pos = 20
    # margin_neg = torch.tensor(0.2)
    margin_pos = torch.tensor(0.8)
    # feats_norm = F.normalize(feats, dim=1, p=2.0)
    # similarity_matrix = torch.matmul(feats_norm, feats_norm.T) # (M, M)
    similarity_matrix = torch.matmul(feats, feats.T).sigmoid() # (M, M)
    # calculate CoSENT_loss
    target_1 = target.clone()
    target_1.fill_diagonal_(0)
    target_1_bool = target_1.to(torch.bool)
    similarity_1 = similarity_matrix * target_1
    target_0 = 1-target
    target_0[torch.where(target_1.sum(-1)==0)]=0
    target_0_bool = target_0.to(torch.bool)
    pos_average = torch.sum(similarity_1, dim=1, keepdim=True) / (torch.sum(target_1, dim=1, keepdim=True)+1e-8) # (M, )
    pos_average[pos_average==0] = 0.8 # if a instance only contains one superpoint OR belongs to the background, we hope it to be a noise
    # neg = torch.exp((similarity_matrix-pos_average+margin_neg) * lambda_neg)
    neg = torch.exp((similarity_matrix-pos_average) * lambda_neg)
    # neg = torch.exp((similarity_matrix-margin_neg) * lambda_neg)
    neg_all = torch.sum(neg[target_0_bool]) / torch.sum(target_0)
    pos = torch.exp((margin_pos-similarity_matrix) * lambda_pos)
    pos_all = torch.sum(pos[target_1_bool]) / torch.sum(target_1)
    loss = torch.log(1 + neg_all + pos_all)
    return loss

def CoSENT_loss_remark_kpos_reg(feats: torch.Tensor, target: torch.Tensor): # feats(M, media), inst_mask(M, M)

    # calculate similarity_matrix
    lambda_neg = 20
    lambda_pos = 20
    lambda_reg = 5
    # margin_neg = torch.tensor(0.2)
    margin_pos = torch.tensor(0.8)
    feats_norm = F.normalize(feats, dim=1, p=2.0)
    similarity_matrix = torch.matmul(feats_norm, feats_norm.T) # (M, M)
    # calculate CoSENT_loss
    target_1 = target.clone()
    target_1.fill_diagonal_(0) 
    target_1_bool = target_1.to(torch.bool)
    similarity_1 = similarity_matrix * target_1
    target_0 = 1-target
    target_0[torch.where(target_1.sum(-1)==0)]=0
    target_0_bool = target_0.to(torch.bool)
    pos_average = torch.sum(similarity_1, dim=1, keepdim=True) / (torch.sum(target_1, dim=1, keepdim=True)+1e-8) # (M, )
    pos_average[pos_average==0] = 0.8 # if a instance only contains one superpoint OR belongs to the background, we hope it to be a noise
    # neg = torch.exp((similarity_matrix-pos_average+margin_neg) * lambda_neg)
    neg = torch.exp((similarity_matrix-pos_average) * lambda_neg)
    neg_all = torch.sum(neg[target_0_bool]) / torch.sum(target_0)
    pos = torch.exp((margin_pos-similarity_matrix) * lambda_pos)
    pos_all = torch.sum(pos[target_1_bool]) / torch.sum(target_1)
    loss = torch.log(1 + neg_all + pos_all)
    loss_reg = torch.relu(-similarity_matrix).mean()
    loss += lambda_reg * loss_reg
    return loss

def infoNCE_loss(feats: torch.Tensor, target: torch.Tensor): # feats(M, media), inst_mask(M, M)
    # calculate similarity_matrix
    temperature = 0.1
    feats_norm = F.normalize(feats, dim=1, p=2.0)
    similarity_matrix = torch.matmul(feats_norm, feats_norm.T) # (M, M)
    similarity_matrix_temperature = similarity_matrix / temperature
    # calculate infoNCE_loss
    target_1 = target.clone()
    target_1.fill_diagonal_(0)
    similarity_1 = similarity_matrix_temperature * target_1
    target_0 = 1-target
    target_0_bool = target_0.to(torch.bool)
    similarity_0 = similarity_matrix_temperature * target_0
    pos_average = torch.sum(similarity_1, dim=1, keepdim=True) / (torch.sum(target_1, dim=1, keepdim=True)+1e-8) # (M, )
    pos_average[pos_average==0] = 0.3 # if a instance only contains one superpoint, we hope it to be a noise
    pos_average_exp = torch.exp(pos_average) #(M,)
    similarity_exp = torch.exp(similarity_0)
    neg_exp_sum = torch.sum(similarity_exp[target_0_bool], dim=1) #(M,)
    proportion = pos_average_exp / (pos_average_exp + neg_exp_sum) #(M,)
    proportion_log = torch.log(proportion)
    loss = -torch.mean(proportion_log)
    return loss

def infoNCE_loss_remark(feats: torch.Tensor, target: torch.Tensor): # feats(M, media), inst_mask(M, M)
    # calculate similarity_matrix
    temperature = 0.1
    feats_norm = F.normalize(feats, dim=1, p=2.0)
    similarity_matrix = torch.matmul(feats_norm, feats_norm.T) # (M, M)
    similarity_matrix_temperature = similarity_matrix / temperature
    # calculate infoNCE_loss
    target_1 = target.clone()
    target_1.fill_diagonal_(0)
    similarity_1 = similarity_matrix_temperature * target_1
    target_0 = 1-target
    pos_average = torch.sum(similarity_1, dim=1, keepdim=True) / (torch.sum(target_1, dim=1, keepdim=True)+1e-8) # (M, )
    pos_average[pos_average==0] = 0.8 # if a instance only contains one superpoint OR belongs to the background, we hope it to be a noise
    pos_average_exp = torch.exp(pos_average) #(M,)
    similarity_exp = torch.exp(similarity_matrix_temperature)
    similarity_exp_0 = similarity_exp * target_0
    neg_exp_sum = torch.sum(similarity_exp_0, dim=1, keepdim=True) #(M,)
    proportion = pos_average_exp / (pos_average_exp + neg_exp_sum) #(M,)
    proportion_log = torch.log(proportion)
    loss = -torch.mean(proportion_log[torch.where(target_1.sum(-1) != 0)])
    return loss



class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_weight):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.register_buffer('cost_weight', torch.tensor(cost_weight))

    @torch.no_grad()
    def forward(self, pred_labels, pred_masks, insts):
        '''
        pred_masks: List[Tensor] len(p2c) == B, Tensor.shape == (n, N)
        pred_labels: (B, n_q, 19)
        insts: List[Instances3D]
        '''
        indices = []
        for pred_label, pred_mask, inst in zip(pred_labels, pred_masks, insts):
            if len(inst) == 0:
                indices.append(([], []))
                continue
            pred_label = pred_label.softmax(-1)  # (n_q, 19)
            tgt_idx = inst.gt_labels  # (num_inst,)
            cost_class = -pred_label[:, tgt_idx]  # (n_q, num_inst)

            tgt_mask = inst.gt_spmasks  # (num_inst, N)

            cost_mask = batch_sigmoid_bce_loss(pred_mask, tgt_mask.float())
            cost_dice = batch_dice_loss(pred_mask, tgt_mask.float())

            C = (self.cost_weight[0] * cost_class + self.cost_weight[1] * cost_mask + self.cost_weight[2] * cost_dice)
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@gorilla.LOSSES.register_module()
class Criterion(nn.Module):

    def __init__(
        self,
        ignore_label=-100,
        loss_weight=[1.0, 1.0, 1.0, 1.0, 1.0],
        cost_weight=[1.0, 1.0, 1.0],
        non_object_weight=0.1,
        num_class=18,
    ):
        super().__init__()
        class_weight = torch.ones(num_class + 1)
        class_weight[-1] = non_object_weight
        self.register_buffer('class_weight', class_weight)
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)
        self.matcher = HungarianMatcher(cost_weight)
        self.num_class = num_class
        self.ignore_label = ignore_label

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_inst_info(self, batched_gt_instance, coords, batch_offsets):
        for i, gt_inst in enumerate(batched_gt_instance):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            coord = coords[start_id:end_id]  # (N, 3)
            inst_idx, point_idx = torch.nonzero(gt_inst['gt_masks'], as_tuple=True)
            inst_point = coord[point_idx]
            gt_inst['gt_center'] = torch_scatter.segment_coo(inst_point, inst_idx.cuda(), reduce='mean')

    def get_layer_loss(self, layer, aux_outputs, insts):
        loss_out = {}
        pred_labels = aux_outputs['labels']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']
        indices = self.matcher(pred_labels, pred_masks, insts)
        idx = self._get_src_permutation_idx(indices)

        # class loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(
            pred_labels.shape[:2],
            self.num_class,
            dtype=torch.int64,
            device=pred_labels.device,
        )  # (B, num_query)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)

        loss_out['cls_loss'] = class_loss.item()

        # # score loss
        score_loss = torch.tensor([0.0], device=pred_labels.device)

        # mask loss
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue
            pred_score = score[idx_q]
            pred_mask = mask[idx_q]  # (num_inst, N)
            tgt_mask = inst.gt_spmasks[idx_gt]  # (num_inst, N)
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss += F.mse_loss(pred_score, tgt_score)
            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            mask_dice_loss += dice_loss(pred_mask, tgt_mask.float())
        score_loss = score_loss / len(pred_masks)
        mask_bce_loss = mask_bce_loss / len(pred_masks)
        mask_dice_loss = mask_dice_loss / len(pred_masks)

        loss_out['score_loss'] = score_loss.item()
        loss_out['mask_bce_loss'] = mask_bce_loss.item()
        loss_out['mask_dice_loss'] = mask_dice_loss.item()

        loss = (
            self.loss_weight[0] * class_loss + self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss + self.loss_weight[3] * score_loss)

        loss_out = {f'layer_{layer}_' + k: v for k, v in loss_out.items()}
        return loss, loss_out

    def forward(self, pred, insts, sp_feats_inst, sample_ids):
        '''
        pred_masks: List[Tensor (n, M)]
        pred_labels: (B, n, 19)
        pred_scores: (B, n, 1) or [(B, n, 1)]
        sp_feats_inst: (B, M, media)
        insts: List[Instance3D]
        '''
        loss_out = {}

        pred_labels = pred['labels']
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # match
        indices = self.matcher(pred_labels, pred_masks, insts)
        idx = self._get_src_permutation_idx(indices)

        # inst_feats_loss
        inst_feats_loss = torch.tensor([0.0], device=pred_labels.device)
        point_p_rates = torch.tensor([0.0], device=pred_labels.device)
        inst_r_rates = torch.tensor([0.0], device=pred_labels.device)
        # for inst in insts:
        #     if len(inst) == 0:
        #         continue
        #     tgt_mask = inst.gt_spmasks  # (num_inst, N)
        #     num_sp_per_inst = tgt_mask.sum(dim=1).unsqueeze(-1) # (num_inst, 1)
        #     sp_feats_inst_expanded = sp_feats_inst.unsqueeze(0) # (1, N, media)
        #     tgt_mask_expanded = tgt_mask.unsqueeze(-1) # (num_inst, N, 1)
        #     inst_feats_expanded = sp_feats_inst_expanded * tgt_mask_expanded # (num_inst, N, media)
        #     # inst_feats = inst_feats_expanded.sum(dim=1) / num_sp_per_inst.expand(-1,sp_feats_inst.size(1)) # (num_inst, media)
        #     inst_feats = inst_feats_expanded.sum(dim=1) / num_sp_per_inst # (num_inst, media)
        #     inst_feats_loss = contrast_loss(inst_feats)

        for sp_feat_inst, inst, sample_id in zip(sp_feats_inst, insts, sample_ids):
            if len(inst) == 0:
                continue
            tgt_mask = inst.gt_spmasks  # (num_inst, N)
            selected = tgt_mask[:,sample_id]
            point_p_rate = torch.mean(torch.sum(selected,dim=0))
            inst_r_rate = torch.mean((torch.sum(selected,dim=1)>0).to(torch.float32))
            tgt_mask_bool = tgt_mask.to(torch.bool)
            tgt_mask_transposed = tgt_mask_bool.T # (N, num_inst)
            target = (tgt_mask_transposed.unsqueeze(0) & tgt_mask_transposed.unsqueeze(1)).sum(dim=2) # (N, N)
            target = target.to(torch.float32)
            # inst_feats_loss = contrast_loss(sp_feats_inst, target)
            inst_feats_loss += CoSENT_loss_remark_kpos(sp_feat_inst, target)
            point_p_rates += point_p_rate
            inst_r_rates += inst_r_rate
        inst_feats_loss = inst_feats_loss / len(pred_masks)
        point_p_rates = point_p_rates / len(pred_masks)
        inst_r_rates = inst_r_rates / len(pred_masks)

        # for sp_feat_inst, inst in zip(sp_feats_inst, insts):
        #     if len(inst) == 0:
        #         continue
        #     tgt_mask = inst.gt_spmasks  # (num_inst, N)
        #     target = tgt_mask
        #     inst_feats_loss = CoSENT_Kpos(sp_feat_inst, target)

        loss_out['inst_feats_loss'] = inst_feats_loss.item()
        loss_out['point_p_rates'] = point_p_rates.item()
        loss_out['inst_r_rates'] = inst_r_rates.item()

        # class loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(
            pred_labels.shape[:2],
            self.num_class,
            dtype=torch.int64,
            device=pred_labels.device,
        )  # (B, num_query)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)

        loss_out['cls_loss'] = class_loss.item()

        # score loss
        score_loss = torch.tensor([0.0], device=pred_labels.device)

        # mask loss
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue
            pred_score = score[idx_q]
            pred_mask = mask[idx_q]  # (num_inst, N)
            tgt_mask = inst.gt_spmasks[idx_gt]  # (num_inst, N)
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss += F.mse_loss(pred_score, tgt_score)
            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            mask_dice_loss += dice_loss(pred_mask, tgt_mask.float())
        score_loss = score_loss / len(pred_masks)
        mask_bce_loss = mask_bce_loss / len(pred_masks)

        loss_out['score_loss'] = score_loss.item()
        loss_out['mask_bce_loss'] = mask_bce_loss.item()
        loss_out['mask_dice_loss'] = mask_dice_loss.item()

        # loss = (
        #     self.loss_weight[0] * class_loss + self.loss_weight[1] * mask_bce_loss +
        #     self.loss_weight[2] * mask_dice_loss + self.loss_weight[3] * score_loss)
        
        loss = (
            self.loss_weight[0] * class_loss + self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss + self.loss_weight[3] * score_loss + self.loss_weight[4] * inst_feats_loss)

        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, insts)
                loss += loss_i
                loss_out.update(loss_out_i)

        loss_out['loss'] = loss.item()

        return loss, loss_out
