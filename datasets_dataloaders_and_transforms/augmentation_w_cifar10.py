import tempfile
from torchvision.datasets import CIFAR10
from torchvision import transforms

cifar10 = CIFAR10(tempfile.gettempdir(),
                  train=True,
                  download=True)

#######################################
######        평균 구하기           ######
#######################################

_mean = cifar10.data.mean(axis=(0, 1, 2)) / 255
_std = cifar10.data.std(axis=(0, 1, 2)) / 255

print('shape   :', cifar10.data[0].shape)
print('RGB mean:', _mean)
print('RGB std :', _std)

# ToTensor를 먼저 호출하여 (c,h,w) 형태로 전달해야 한다.
aug_f = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(_mean, _std)])
img = aug_f(cifar10.data[0])

print('augmented img shape:', img.shape)
print('augmented img mean :', img.mean(axis=(1, 2)))
print('augmented img std  :', img.std(axis=(1, 2)))

import matplotlib.pyplot as plt
from PIL import Image

def display_augmented_images(aug_f):
    fig, subplots = plt.subplots(2, 5, figsize=(13, 6))
    for i in range(5):
        axi1 = subplots.flat[i]
        axi2 = subplots.flat[i+5]

        original_img = Image.fromarray(cifar10.data[i])
        augmented_img = aug_f(original_img)

        axi1.imshow(original_img)
        axi2.imshow(augmented_img)
        axi1.set_title('original_img')
        axi2.set_title('augmented_img')

#######################################
###      Random Horizontal Flip     ###
#######################################
# p : flip을 적용할 확률
flip = transforms.RandomHorizontalFlip(p=1)
display_augmented_images(flip)

#######################################
###      Random Horizontal Flip     ###
#######################################
# p : flip을 적용할 확률
aug_f = transforms.RandomVerticalFlip(p=1)
display_augmented_images(aug_f)

#######################################
#          Random Affine            
#######################################
# Affine transformation : 
#       - points, straight lines, and planes을 보존하는 linear mapping.
#       - A Combination of Translation, Rotatoin, Shear, Scale
# torchvision.transforms.RandomAffine(degrees, translate=None, 
#               scale=None, shear=None, resample=0, fillcolor=0)
aug_f = transforms.RandomAffine(degrees=30)
display_augmented_images(aug_f)

#######################################
#            Random Crop            
#######################################
# (20, 20 size로 return)
aug_f = transforms.RandomCrop((20, 20))
display_augmented_images(aug_f)

#######################################
#        Random Resized Crop            
#######################################
#torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), 
# ratio=(0.75, 1.3333333333333333), interpolation=2)
# random size (0.8~1) 와 random aspect ratio (3/4 ~ 4/3)으로 crop 후 resize
aug_f = transforms.RandomResizedCrop((22, 22)) # <- output size : 22,22
display_augmented_images(aug_f)

#######################################
#          Random Gray Scale           
#######################################
# p : gray scale 로 바꿀 확률
aug_f = transforms.RandomGrayscale(p=1)
display_augmented_images(aug_f)

#######################################
#         Random Perspective       
#######################################

aug_f = transforms.RandomPerspective()
display_augmented_images(aug_f)

#######################################
#          Random Rotation          
#######################################
# range of degrees : (-degrees, degrees)
# expand : True 일 경우 회전 후에도 전체 이미지가 나옴. 즉, 모서리가 잘리지 않음
aug_f = transforms.RandomRotation(degrees=90, expand=False)
display_augmented_images(aug_f)

#######################################
#            Random Choice           
#######################################
# list로 부터 하나의 transformation 만 골라서 적용
aug_f = transforms.RandomChoice([transforms.RandomGrayscale(p=1), 
                                 transforms.RandomVerticalFlip(p=1)])
display_augmented_images(aug_f)

#######################################
#          Color Jitter (ALL)            
#######################################
# 하나씩 따로 사용 가능
aug_f = transforms.ColorJitter(brightness=(0.2, 2), 
                               contrast=(0.3, 2), 
                               saturation=(0.2, 2), 
                               hue=(-0.3, 0.3))
display_augmented_images(aug_f)


#######################################
#              Resize            
#######################################
# size 만큼 resize
aug_f = transforms.Resize(size=(100, 100))
display_augmented_images(aug_f)


#######################################
#          Channel transpose            
#######################################

data = cifar10.data
data = torch.Tensor(data)
print('data shape:', data.shape)
print('permute   :', data.permute(0, 3, 1, 2).shape)
print('transpose :', data.transpose(1, 3).shape)

data shape: torch.Size([50000, 32, 32, 3])
permute   : torch.Size([50000, 3, 32, 32])
transpose : torch.Size([50000, 3, 32, 32])

#######################################
#              To Tensor            
#######################################

img = cifar10.data[0]
aug_f = transforms.ToTensor()

print('image shape          :', img.shape)
print('augmented image shape:', aug_f(img).shape)

image shape          : (32, 32, 3)
augmented image shape: torch.Size([3, 32, 32])