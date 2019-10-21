import cv2
from matplotlib import pyplot as plt

img = cv2.imread("/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/rgb/0/77.png")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray_edge = cv2.Canny(img_gray,100,200)
plt.imshow(img_gray,cmap="gray")
plt.imshow(img_gray_edge,cmap="gray")
plt.xticks([]), plt.yticks([])
plt.show()
# cv2.imshow(img[:,:,0],cmap="gray")
# cv2.waitKey(0)
# cv2.destroyAllWindows()
