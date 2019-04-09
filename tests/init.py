import face_recognition
import cv2

input_movie = cv2.VideoCapture()

# 初始化一些变量
face_locations = []
face_encodings = []
face_names = []

flag = input_movie.open("./source/video/speaks.mp4")
if flag is False:
    print("Can't open file")

while True:
    ret, frame = input_movie.read()
    # frame_number += 1
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # 视频结束，自动退出
    if not ret:
        break
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) is not 0:
        print(face_locations)
    # Label the results
    for (top, right, bottom, left) in face_locations:
        print("Draw")
        # top *= 4
        # right *= 4
        # bottom *= 4
        # left *= 4
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)

    cv2.imshow('frame', frame)

    # 延迟
    key = cv2.waitKey(delay=30)
    if key == ord("q"):
        break

input_movie.release()
cv2.destroyAllWindows()
