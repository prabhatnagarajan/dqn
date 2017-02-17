import imageio
import os
images = []
count = 0
for filename in os.listdir("recordings"):
    print count
    if filename.endswith(".png"):
        images.append(imageio.imread("recordings/" +filename))
    count = count + 1
    if count == 100:
        break
imageio.mimsave('movie.gif', images)