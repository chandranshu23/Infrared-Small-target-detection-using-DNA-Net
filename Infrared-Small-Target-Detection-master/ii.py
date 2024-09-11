import os
import numpy as np 
from PIL import Image
a = os.path.exists("C:/Users/chand/Desktop/Project 2/DNA-Net/Infrared-Small-Target-Detection-master/dataset/MWIRSTD/masks/6-0358-image_masks.jpg")
print(a)
b = os.path.exists("C:/Users/chand/Desktop/Project 2/DNA-Net/Infrared-Small-Target-Detection-master/dataset/MWIRSTD/masks/6-0358-image_masks.png")
print(b)
aa = Image.open("C:/Users/chand/Desktop/Project 2/DNA-Net/Infrared-Small-Target-Detection-master/dataset/MWIRSTD/masks/6-0358-image_masks.png")
aa = np.array(aa, dtype=np.uint8)
for i in range (0,8):
    print(aa[160+i][265+i])
    print(aa[265+i][160+i])
#print(aa[160:167][265:271])
#print(aa[265:271][160:167])
aa = (aa == 1)*aa
print(aa.shape)
#print(aa)
##for k in range(0, aa.shape[2]):
#for i in range(0,aa.shape[0]):
#    for j in range(0,aa.shape[1]):
#        if aa[i][j] == 1.0:
#            aa[i][j]= 255
#        else:
 #           aa[i][j]=0
print("hello")
for i in range (0,8):
    print(aa[160+i][265+i])
    print(aa[265+i][160+i])

print(aa.shape)
a = np.zeros([2,2])
print(a.shape)
bb = np.expand_dims(a, axis=0).astype('float32')
print(bb.shape)