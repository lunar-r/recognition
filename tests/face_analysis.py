import face_recognition


img = face_recognition.load_image_file("people.jpg")

locations = face_recognition.face_locations(img)

landmarks_list = face_recognition.face_landmarks(img)
