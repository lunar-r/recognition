import face_recognition


img = face_recognition.load_image_file("../source/images/Ariel_Sharon/Ariel_Sharon_0001.jpg")
location = face_recognition.face_locations(img)
print(location)