import pandas as pd
import numpy as np
import os
folder_root = '/root/FEDformer/results/Final_FEDformer_random_modes64_custom_ftM_sl12_ll6_pl2_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0'
data = np.load(os.path.join(folder_root, 'pred.npy'))
data_2 = np.load(os.path.join(folder_root, 'true.npy'))
print(data.shape)
# np.savetxt('pred_csv.csv', data[:,0,-1], delimiter=',')
np.savetxt('true_csv.csv', data_2[:,0,:], delimiter=',')


# Tạo dữ liệu mẫu cho cột giá trị
values = data[:,0,-1]

start_time = pd.to_datetime('2023-02-16 20:00:00.000')
end_time = pd.to_datetime('2023-06-01 01:00:00.000')

# Tạo chuỗi thời gian với tần số là 1 giờ
time_index = pd.date_range(start=start_time, end=end_time, freq='H')

# Nếu số lượng giá trị không đủ, duplicate 2 giá trị cuối cùng
while(1):
    if len(values) < len(time_index):
        last_values = values[-1:]
        # print(last_values.shape)
        # print(values.shape)
        values = np.concatenate([values,last_values])
        # print(len(values))
    else:
        break

# Chuyển đổi thời gian sang timestamp với đơn vị là milisecond
# timestamp = time_index.astype(np.int64) // 10**6  # Chia cho 10^6 để chuyển đổi sang milisecond
from datetime import datetime
timestamp = []
for i in range(len(time_index)):
    dt_obj = datetime.strptime(str(time_index[i]),
                            '%Y-%m-%d %H:%M:%S')
    timestamp_i = dt_obj.timestamp() * 1000
    timestamp.append(timestamp_i)
timestamp = np.asarray(timestamp)

# Tạo DataFrame với hai cột
df = pd.DataFrame({'OPEN_TIME': timestamp, 'PREDICTION': values})

# Lưu DataFrame vào file CSV
df.to_csv('data.csv', index=False)