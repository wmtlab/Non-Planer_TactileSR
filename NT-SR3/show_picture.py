from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 图片路径

image_paths = [lr, hr, sr]

# 文本标识
labels = ['4x4', 'ground true', 'S R']


# 读取图片
img1 = mpimg.imread(lr)
img2 = mpimg.imread(hr)
img3 = mpimg.imread(sr)
# 创建一个新的图形
plt.figure()

# 添加第一个子图
plt.subplot(1, 3, 1)  # (rows, columns, panel number)
plt.imshow(img1)
plt.title(labels[0])
plt.axis('off')  # 关闭坐标轴

# 添加第二个子图
plt.subplot(1, 3, 2)
plt.imshow(img2)
plt.title(labels[1])
plt.axis('off')


# 添加第二个子图
plt.subplot(1, 3, 3)
plt.imshow(img3)
plt.title(labels[2])
plt.axis('off')
# 显示图形
plt.show()
