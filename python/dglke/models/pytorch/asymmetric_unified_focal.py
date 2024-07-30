import torch


# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5:
        return [2, 3, 4]

    # Two dimensional
    elif len(shape) == 4:
        return [2, 3]

    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


def asymmetric_focal_loss(inputs: torch.Tensor, target, delta=0.7, gamma=2., epsilon=1e-07):
    inputs = torch.clamp(inputs, epsilon, 1. - epsilon)
    targets = torch.tile(torch.tensor(target).float().unsqueeze(-1), inputs.shape).cuda()
    cross_entropy = -targets * torch.log(inputs)
        
    # Calculate losses separately for each class, only suppressing background class
    back_ce = torch.pow(1 - inputs[:,0,:,:], gamma) * cross_entropy[:,0,:,:]
    back_ce =  (1 - delta) * back_ce

    fore_ce = cross_entropy[:,1,:,:]
    fore_ce = delta * fore_ce

    loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

    return loss

def asymmetric_focal_tversky_loss(inputs: torch.Tensor, target, delta=0.7, gamma=2., epsilon=1e-07):
    # Clip values to prevent division by zero error
    inputs = torch.clamp(inputs, epsilon, 1. - epsilon)
    targets = torch.tile(torch.tensor(target).float().unsqueeze(-1), inputs.shape).cuda()
    axis = identify_axis(targets.size())

    # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
    tp = torch.sum(targets * inputs, axis=axis)
    fn = torch.sum(targets * (1-inputs), axis=axis)
    fp = torch.sum((1-targets) * inputs, axis=axis)
    dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

    # Calculate losses separately for each class, only enhancing foreground class
    back_dice = (1-dice_class[:,0]) 
    fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -gamma) 

    # Average class scores
    loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
    return loss

def asymmetric_unified_focal_loss(inputs: torch.Tensor, target, weight=0.5, delta=0.6, gamma=0.2):
    asymmetric_ftl = asymmetric_focal_tversky_loss(inputs=inputs, target=target, delta=delta, gamma=gamma)
    asymmetric_fl = asymmetric_focal_loss(inputs=inputs, target=target, delta=delta, gamma=gamma)

    # Return weighted sum of Asymmetrical Focal loss and Asymmetric Focal Tversky loss
    if weight is not None:
        return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)  
    else:
        return asymmetric_ftl + asymmetric_fl