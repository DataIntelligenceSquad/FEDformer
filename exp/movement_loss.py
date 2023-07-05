import torch
import torch.nn as nn

class MovementLoss(nn.Module):
    def __init__(self):
        super(MovementLoss, self).__init__()

    def forward(self, predictions, targets):
        # Tính toán sự thay đổi giữa các giá trị thực tế
        true_changes = targets[:,1:,:] - targets[:,:-1,:]
        
        # Tính toán sự thay đổi của các giá trị dự đoán so với giá trị thực tế
        pred_changes = predictions[:,1:,:] - targets[:,:-1,:]
        
        # Tính toán hàm loss dựa trên sự khác biệt giữa true_changes và pred_changes
        loss = torch.mean(torch.abs(true_changes - pred_changes))
        
        return loss