from PIL import Image
import numpy as np

path="/mnt/large/pentalo/deep/external_datasets/princeton/EvaluationSet/bag1/depth/d-0-1.png"
im = Image.open(path)
red = im.getextrema()

print(red)

im.show()


