import face_recognition
import cv2
import sys
input_movie = cv2.VideoCapture(0)

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# 创建输出文件
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))

# 先加载进需要识别的人
JK_image = face_recognition.load_image_file("./source/img/Joanne.jpg")
jk_face_encoding = face_recognition.face_encodings(JK_image)[0]
XG_image = face_recognition.load_image_file("./source/img/xg.jpg")
xg_face_encoding = face_recognition.face_encodings(XG_image)[0]
person = face_recognition.load_image_file("./source/img/stanford/stanford_20002.jpg")
person_encoding = face_recognition.face_encodings(person)[0]
known_faces = [
    jk_face_encoding,
    xg_face_encoding
]

# 初始化一些变量
face_locations = []
face_encodings = []
face_names = []
# frame_number = 0


process_frame = True


while True:
    ret, frame = input_movie.read()
    # frame_number += 1
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # 视频结束，自动退出
    if not ret:
        break
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = small_frame[:, :, ::-1]

    process_frame = not process_frame
    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        if not name:
            continue
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('frame', frame)

    # 结果写入文件
    # print("Writing frame {} / {}".format(frame_number, length))
    # output_movie.write(frame)

    # 延迟
    key = cv2.waitKey(delay=20)
    if key == ord("q"):
        break

input_movie.release()
cv2.destroyAllWindows()
