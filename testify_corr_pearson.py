import torch
import numpy as np
from scipy.stats import pearsonr

# 1. 模拟 3 个脑区的信号，每个信号 1000 个时间点
np.random.seed(42)
signals = np.random.rand(1000, 3) # 形状: (时间, 脑区)

# --- 方法 A: 你的代码 (torch.corrcoef) ---
# 注意你的代码里有 .T，是因为 corrcoef 默认每一行是一个变量
signals_tensor = torch.tensor(signals.T, dtype=torch.float)
fc_matrix = torch.corrcoef(signals_tensor).numpy()

# --- 方法 B: 手动逐一计算 Pearson ---
# 计算脑区 0 和 脑区 1 的相关性
r_val, p_val = pearsonr(signals[:, 0], signals[:, 1])

# 2. 验证
print(f"矩阵中 [0,1] 的值: {fc_matrix[0, 1]:.6f}")
print(f"Pearsonr 计算的值: {r_val:.6f}")

# 检查差值
diff = np.abs(fc_matrix[0, 1] - r_val)
print(f"两者差值: {diff:.2e}")