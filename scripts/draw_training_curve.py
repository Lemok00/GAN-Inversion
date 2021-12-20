import matplotlib.pyplot as plt
import numpy as np

exp_name = 'resnet34_FFHQ256_steps3'

training_log_path = f'../experiments/{exp_name}/training_logs.txt'

with open(training_log_path, 'r') as f:
    lines = f.readlines()
    loss_lines = [lines[i] for i in range(1, len(lines), 2)]

total_losses = []
recon_losses = []
lpips_losses = []

for loss_line in loss_lines:
    split = loss_line.split(' ')
    total_losses.append(float(split[1]))
    recon_losses.append(float(split[3]))
    lpips_losses.append(float(split[5]))

total_losses = np.array(total_losses)
recon_losses = np.array(recon_losses)
lpips_losses = np.array(lpips_losses) * 0.8

iters = np.array(range(1, len(total_losses) * 100, 100))

plt.figure(figsize=(6, 6))

plt.plot(iters, total_losses, color='r', lw=1, linestyle='-', label=f'Total Loss')
plt.plot(iters, recon_losses, color='b', lw=1, linestyle='-', label=f'L2 Loss')
plt.plot(iters, lpips_losses, color='g', lw=1, linestyle='-', label=f'LPIPS Loss')

plt.xticks(np.linspace(0, 80000, 9))
plt.yticks(np.linspace(0, 1, 11))
plt.grid()
plt.xlabel('Loss')
plt.ylabel('Iteration')
# plt.title('The ROC curve of StegExpose')
plt.legend(loc="upper right")
plt.savefig(f'../experiments/{exp_name}/training_loss.png')
plt.show()
