import cv2
import numpy as np
from matplotlib import pyplot as plt
import math







def create_attention_filter(width,height,channel,p = 1):
    half_height = math.floor(height/2)
    half_width = math.floor(width/2)
    half_height_filter = [x / height for x in range(1, half_height+1)]
    half_width_filter = [x / width for x in range(1, half_width+1)]
    filter = np.zeros((height,width))

    for i in range(half_height):
        for j in range(half_width):
            filter[height -i - 1][width - j -1] = filter[height -i - 1][j] = filter[i][width - j -1] = filter[i][j] = half_height_filter[i] + half_width_filter[j]

    filter = np.power(filter, p)

    result = np.zeros((height,width,channel))
    for h in range(height):
        for w in range(width):
            for c in range(channel):
                result[h,w,c] = filter[h,w]
    return result

def create_attention_filter(width,height,channel,p = 1,is_model=False):
    half_height = math.floor(height/2)
    half_width = math.floor(width/2)
    half_height_filter = [x / height for x in range(1, half_height+1)]
    half_width_filter = [x / width for x in range(1, half_width+1)]
    filter = np.zeros((height,width))

    for i in range(half_height):
        for j in range(half_width):
            filter[height -i - 1][width - j -1] = filter[height -i - 1][j] = filter[i][width - j -1] = filter[i][j] = half_height_filter[i] + half_width_filter[j]

    filter = np.power(filter, p)
    if is_model:
        return np.array([filter for i in range(channel)])
    result = np.zeros((height,width,channel))
    for h in range(height):
        for w in range(width):
            for c in range(channel):
                result[h,w,c] = filter[h,w]
    return result

def apply(img,attention_filter):
    return np.ceil(img * attention_filter).astype("uint8")



if __name__ == '__main__':
    img = cv2.imread("/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/rgb/0/29.png")
    attention_filter = create_attention_filter(32,32,3)




    # img_att = apply(img,attention_filter)

    plt.xticks([]), plt.yticks([])
    plt.imshow(img)
    plt.show()
    plt.imshow(apply(img,create_attention_filter(32,32,3)))
    plt.show()
    plt.imshow(apply(img,create_attention_filter(32,32,3,2)))
    plt.show()
    plt.imshow(apply(img,create_attention_filter(32,32,3,3)))
    plt.show()
    # cv2.imshow(img[:,:,0],cmap="gray")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
