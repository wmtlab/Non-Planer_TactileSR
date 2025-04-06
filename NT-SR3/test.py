import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# # 假设 original_image 是原始的4x4图像，ground_truth 是对应的40x40真实图像
# original_image = cv2.imread('/home/hedazhong/zhou_project/diff_SR/experiments/sr_ffhq_240516_173805/results/3667/110000_1_lr.png', cv2.IMREAD_GRAYSCALE)
# ground_truth = cv2.imread('/home/hedazhong/zhou_project/diff_SR/experiments/sr_ffhq_240516_173805/results/3667/110000_1_hr.png', cv2.IMREAD_GRAYSCALE)
#
# # 双线性插值
# interpolated_image = cv2.resize(original_image, (40, 40), interpolation=cv2.INTER_LINEAR)
original_image = cv2.imread('/home/hedazhong/zhou_project/diff_SR/experiments/sr_ffhq_240516_173805/results/1334/40000_3_hr.png', cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread('/home/hedazhong/zhou_project/diff_SR/experiments/sr_ffhq_240516_173805/results/1334/40000_3_sr.png', cv2.IMREAD_GRAYSCALE)
# 计算SSIM
ssim_value = ssim(ground_truth, original_image, full=False)

# 计算PSNR
psnr_value = cv2.PSNR(ground_truth, original_image)

# 打印SSIM和PSNR值
print(f"SSIM: {ssim_value}")
print(f"PSNR: {psnr_value}")
