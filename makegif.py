import imageio
import os
images = []
count = 0
for filename in os.listdir("recordings"):
    print count
    if filename.endswith(".png"):
    	for _ in range(3):
        	images.append(imageio.imread("recordings/" +filename))
    count = count + 1
    if count == 200:
        break
imageio.mimsave('movie.gif', images)