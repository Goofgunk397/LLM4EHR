import torch
import torch.nn as nn
import torch.nn.functional as F


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class PatchNCELoss(nn.Module):
    """
    Note that the patch mask is still included here since we cant allow empty patches to match against eachother
    Temporal masking is uncessary here
    """
    def __init__(self, temperature=0.02, temporal_masking=False):
        super().__init__()
        self.tau = temperature

    def forward(self, ts_out, seq_out, omega, patch_mask):
        patch_mask = patch_mask.float().detach()
        
        masked_omega = torch.eye(patch_mask.shape[0], device=patch_mask.device)
        masked_omega = masked_omega*patch_mask
        patch_sum = masked_omega.sum() + 1e-6   # EHR sequence can be sparse so an eps is needed here
        ts_out = F.normalize(ts_out.reshape(-1, ts_out.shape[-1]), dim=-1)
        seq_out = F.normalize(seq_out.reshape(-1, seq_out.shape[-1]), dim=-1)
        ts_dot_seq = torch.div(ts_out @ seq_out.T, self.tau)
        # for stability
        logits_max, _ = torch.max(ts_dot_seq, dim=1, keepdim=True)
        logits = ts_dot_seq - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = masked_omega*(logits - torch.log(exp_logits.sum(dim=1, keepdim=True)))
        mean_omega_log_prob = -log_prob.sum()/patch_sum

        return mean_omega_log_prob
    
class PatchSoftLoss(nn.Module):
    def __init__(self, temperature=0.02, temporal_masking=False):
        super().__init__()
        self.tau = temperature
        self.temporal_masking = temporal_masking  

    def forward(self, ts_out, seq_out, omega, patch_mask):
        patch_mask = patch_mask.float().detach()
        if self.temporal_masking:
            patch_mask = torch.tril(patch_mask)
        patch_sum = patch_mask.sum()        
        masked_omega = F.softmax(omega*patch_mask, dim=-1)
        ts_out = F.normalize(ts_out.reshape(-1, ts_out.shape[-1]), dim=-1)
        ts_dot_ts = torch.div(ts_out @ ts_out.T, self.tau)
        # for stability
        logits_max, _ = torch.max(ts_dot_ts, dim=1, keepdim=True)
        logits = ts_dot_ts - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = masked_omega*(logits - torch.log(exp_logits.sum(dim=1, keepdim=True)))
        mean_omega_log_prob = -log_prob.sum()/patch_sum

        return mean_omega_log_prob

class PatchContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.02, temporal_masking=False):
        super().__init__()
        self.nce_loss = PatchNCELoss(temperature=temperature, temporal_masking=temporal_masking)
        self.soft_loss = PatchSoftLoss(temperature=temperature, temporal_masking=temporal_masking)

    def forward(self, ts_out, seq_out, omega, patch_mask):
        l1 = self.nce_loss(ts_out, seq_out, omega, patch_mask)
        l2 = self.soft_loss(ts_out, seq_out, omega, patch_mask)

        return l1, l2


class PatchReconLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ts_pred, ts_in):
        mse_loss = F.mse_loss(ts_pred, ts_in, reduction='mean')
        return mse_loss

