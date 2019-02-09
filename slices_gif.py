import numpy as np
import imageio
#import torch
#import torch.nn.functional as F
image_size =64
version = 7
image = np.load(f"featureCube{image_size}_{version}.npy")[0]
#image = image.transpose(3,0,1,2) #b,c,a,d
#with torch.no_grad():
#    image = F.upsample(torch.Tensor(image).unsqueeze(0),scale_factor = (2,2,2), mode="trilinear")
#    image = image.squeeze(0).numpy().transpose(1,2,3,0)

image = (image*255).astype('uint8')
print(image.shape)
image = np.concatenate([image,
                       np.transpose(image,[1,0,2,3]),
                       np.transpose(image,[2,1,0,3])], 2)
to_show = []
for i in range(image.shape[0]):
    to_show.append(np.flip(image[i],2))
imageio.mimsave("3dvis_2.gif", to_show,fps=20)
