import torch
import torch.nn as nn

class MovementLoss(nn.Module):
    def __init__(self):
        super(MovementLoss, self).__init__()

    def forward(self, predictions, targets, pre_batch_y):
        # print(type(pre_batch_y))
        if not isinstance(pre_batch_y, int):
            first_true_change = targets[:, :1, :] - pre_batch_y[:, -1:, :]
            first_pred_change = predictions[:, :1, :] - pre_batch_y[:, -1:, :]
        else:
            first_true_change = targets[:, :1, :] - targets[:, -1:, :]
            first_pred_change = predictions[:, :1, :] - targets[:, -1:, :]
        
        # Tính toán sự thay đổi giữa các giá trị thực tế
        true_changes = targets[:, 1:, :] - targets[:, :-1, :]
        
        # Tính toán sự thay đổi của các giá trị dự đoán so với giá trị thực tế
        pred_changes = predictions[:, 1:, :] - predictions[:, :-1, :]

        true_changes = torch.cat((first_true_change, true_changes), dim = 1)
        pred_changes = torch.cat((first_pred_change, pred_changes), dim = 1)
        
        # Tạo tensor chứa các giá trị 0 và 1 tương ứng với các trường hợp dấu trái ngược và dấu cùng hướng
        sign_diff = torch.sign(true_changes * pred_changes)
        # print("tensor: ", true_changes * pred_changes)
        # print("sign_diff: ", sign_diff)
        tmp_movement_loss = torch.abs(true_changes - pred_changes) * sign_diff.float()
        positive_indices = torch.gt(tmp_movement_loss, torch.zeros_like(tmp_movement_loss))
        # Gán các giá trị dương thành 0
        tmp_movement_loss[positive_indices] = 0
        tmp_movement_loss = torch.abs(tmp_movement_loss)
        # print("tmp_movement_loss: ", tmp_movement_loss)
        # Tính toán hàm loss dựa trên sự khác biệt giữa true_changes và pred_changes, với trọng số tăng cao cho các giá trị trái dấu nhau
        loss = torch.mean(tmp_movement_loss)
        
        return loss