import torch
import torch.nn as nn


class CoherenceLoss(nn.Module):
    """
    Compute the Unsupervised Coherence Loss

    PARAMETERS
    ----------
    global_weight: float
        Global weight to apply on the entire loss
    """

    def __init__(self, global_weight: float = 1.) -> None:
        super(CoherenceLoss, self).__init__()
        self.global_weight = global_weight
        self.mse = nn.MSELoss()

    def forward(self, rd_input: torch.Tensor,
                ra_input: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the loss between the two predicted view masks"""
        rd_softmax = nn.Softmax(dim=1)(rd_input)
        ra_softmax = nn.Softmax(dim=1)(ra_input)

        # get range prob from RD map 
        rd_range_probs = torch.max(rd_softmax, dim=3, keepdim=True)[0]
        print("rd_range_probs shape : ",rd_range_probs.shape)
        print("rd_range_probs : ",rd_range_probs)
        rd_range_probs = torch.rot90(rd_range_probs, 2, [2, 3]) # Rotate RD Range vect to match zero range
        print("rot rd_range_probs shape : ",rd_range_probs.shape)
        print("rot rd_range_probs : ",rd_range_probs)
        # get range prob from RA map 
        ra_range_probs = torch.max(ra_softmax, dim=3, keepdim=True)[0]
        print("ra_range_probs shape : ",ra_range_probs.shape)
        print("ra_range_probs : ",ra_range_probs)
        # mse from RD range to RA range
        coherence_loss = self.mse(rd_range_probs, ra_range_probs)
        
        # return the weighted loss
        weighted_coherence_loss = self.global_weight * coherence_loss
        return weighted_coherence_loss


if __name__ == "__main__":
    # 假设 targets 是整数标签
    RDlogits = torch.randint(0, 4, (1, 3, 4, 3)).float().cuda()

    RAlogits = torch.randint(0, 4, (1, 3, 4, 4)).float().cuda()


    loss_fn = CoherenceLoss(global_weight=1.).cuda()
    loss = loss_fn(RDlogits, RAlogits)
    print("focal loss : ",loss)
