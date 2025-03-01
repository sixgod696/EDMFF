import torch
import torch.nn.functional as F
from utils.AF.Fsmish import smish as Fsmish

def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss modified in DexiNed

    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum

    # print("---------------bdcnloss权重-------------------")
    # print(l_weight)
    # print(cost)
    # print(l_weight*cost)
    # print("---------------bdcnloss权重-------------------")
    return l_weight*cost

# ------------ cats losses ----------
def bdrloss(prediction, label, radius,device='cpu'):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt.requires_grad = False
    filt = filt.to(device)

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)

    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return torch.sum(cost.float().mean((1, 2, 3)))

def textureloss(prediction, label, mask_radius, device='cpu'):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.to(device)
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
    filt2.requires_grad = False
    filt2 = filt2.to(device)

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss.float().mean((1, 2, 3)))

#修改添加Dice_loss
def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    #print(inputs.shape)#torch.Size([8, 1, 300, 300])
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_inputs = temp_inputs.reshape(8,300,300)
    temp_target = target.view(n, -1, ct)
    #print(temp_inputs.shape)#torch.Size([8, 90000, 1])
    #print(temp_target.shape)#torch.Size([8, 300, 300])
    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#

    #tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    tp = torch.sum(temp_target[..., :] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    #fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def cats_loss(prediction, label, l_weight=[0.,0.], device='cpu'):
    # tracingLoss

    tex_factor,bdr_factor = l_weight

    #tex_factor = 0.01
    #bdr_factor = 3.0
    
    tex_factor = 0.01
    bdr_factor = 3.0

    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0

    prediction = torch.sigmoid(prediction)

    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduction='none')
    cost = torch.sum(cost.float().mean((1, 2, 3)))  # by me
    label_w = (label != 0).float()
    #修改添加loss
    diceloss = Dice_loss(prediction.float(), label_w.float())

    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4, device=device)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4, device=device)
    # print("---------------Catsloss权重-------------------")
    # # #print(bdr_factor)#3
    # # #print(tex_factor)#0.01
    # # #print(diceloss)
    # # #print(cost + bdr_factor * bdrcost + tex_factor * textcost)
    # print(cost + bdr_factor * bdrcost + tex_factor * textcost + 0.1 * diceloss)
    # print("---------------Catsloss权重-------------------")
    #return cost + bdr_factor * bdrcost + tex_factor * textcost + diceloss
    return cost + bdr_factor * bdrcost + tex_factor * textcost