import glob
import skimage as sk
import random
from skimage import transform,io,filters

foldernames = glob.glob("images/Folio/*")

for folder in foldernames:
    filenames = glob.glob(folder+"/*.jpg")

    count = 0
    for img in filenames:
        xmg = sk.io.imread(img) #read image
        
        #rotate
        random_degree = random.uniform(-25,25)
        rotated_image = transform.rotate(xmg, random_degree,cval=1, mode="constant")
        rotate_file_path = '%s/rotated_%s.jpg' % (folder, count)
        sk.io.imsave(rotate_file_path, rotated_image)
        
        #noise
        noised_image = sk.util.random_noise(xmg)
        noised_file_path = '%s/noised_%s.jpg' % (folder, count)
        sk.io.imsave(noised_file_path, noised_image)
        
        #distorted
#         tf = transform.AffineTransform(rotation=0.1, shear=0.2)
#         distorted_image = transform.warp(xmg,tf)
#         distorted_path = '%s/distorted_%s.jpg' % (folder, count)
#         sk.io.imsave(distorted_path, distorted_image)
        
        #blur
        blur_image = filters.gaussian(xmg,sigma=20,mode="reflect")
        blur_path = '%s/blur_%s.jpg' % (folder, count)
        sk.io.imsave(blur_path, blur_image)
        
        count += 1
