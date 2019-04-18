# from mtcnn.mtcnn import MTCNN
# import cv2
# img = cv2.imread("../source/otherImg/people.jpg")
#
# detector = MTCNN()
#
# detector.detect_faces()
import cv2
from deep.dnn import DeepFace

work = DeepFace()
img = cv2.imread("./source/images/Donald_Trump/Donald_Trump_1013.jpg", 1)
print(img)
align = img[...,::-1]
align = work.align_image(img)


cv2.imshow("align", align)
cv2.waitKey(0)
