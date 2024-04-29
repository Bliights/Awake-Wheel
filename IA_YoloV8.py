import os, random
from ultralytics import YOLO
from PIL import Image
import cv2

HOME = os.getcwd()

model = YOLO("modelV1.pt")

#choose random test set image

test_set_loc = f'{HOME}/datasets/Expression-Recognition-2/test/images/'
random_test_image = random.choice(os.listdir(test_set_loc))
print("running inference on " + random_test_image)

pred = model.predict(source = test_set_loc + random_test_image)
pred
pred = model.predict(source = f"{HOME}\\datasets\\Expression-Recognition-2\\train\\images\\1_jpg.rf.90fbc79a9c7174d528ecd41689199d09.jpg")
pred

