import numpy as np
from scipy.ndimage import shift

width = 4
height = 3
depth = 2

img = np.zeros((height, width, depth), dtype=np.int16)

for d in range(depth):
    for i in range(height):
        for j in range(width):
            img[i][j][d] = (d+1)*10+j

def print_image(img):
    print(img.shape)
    depth = img.shape[2]
    for d in range(depth):
        print(img[:,:,d])
print(".........................................")
print_image(img)
print(".........................................")
def transform(img, tx):
    #nueva imagen ej: columna de ceros + img[0:n-1]
    s = 0
    if tx > 1.0 or tx < -1.0:
        s = 0
    else:
        s = int(tx*width)

    return np.stack(tuple(shift(img[:,:,i], s, mode='nearest') for i in range(depth)), axis=-1)


xs = transform(img, 0.5) #a number [-1.0, 1.0]
print_image(xs)
print(".........................................")
xs = transform(img, -0.3) #a number [-1.0, 1.0]
print_image(xs)
print(".........................................")
xs = transform(img, -1.3) #a number [-1.0, 1.0]
print_image(xs)
print(".........................................")
