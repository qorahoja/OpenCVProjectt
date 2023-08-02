import cv2  # Suratlarni olish uchun OpenCV kutubxonasini import qilamiz
import numpy as np  # Ma'lumotlar bilan ishlash uchun Numpy kutubxonasini import qilamiz
import csv  # CSV fayllar bilan ishlash uchun CSV kutubxonasini import qilamiz
import pandas as pd  # Ma'lumotlar tahlili uchun Pandas kutubxonasini import qilamiz
import random

def search_csv(file_path, name, surname, age):
    # CSV faylda ma'lumotlarni qidirish uchun funksiya
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)  # CSV faylni o'qish obyekti yaratamiz
        for row in csv_reader:  # CSV faylni qatorlar bo'yicha o'qiyap chiqamiz
            if len(row) >= 3 and row[0] == name and row[1] == surname and int(row[2]) == age:
                return row  # Ma'lumot topilganda uni qaytarib chiqamiz
    return None  # Ma'lumot topilmaganda None qaytarib chiqamiz

def sorovnoma():
    # Foydalanuvchidan ma'lumotlarni so'raydigan funksiya
    name = input("Ismingizni kiriting: ")  # Foydalanuvchidan ismni so'raymiz
    surname = input("Familyangizni kiriting: ")  # Foydalanuvchidan familiyani so'raymiz
    age = int(input("Yoshingizni kiriting: "))  # Foydalanuvchidan yoshni so'raymiz

    file_path = "user_info.csv"  # CSV faylning yo'lini belgilaymiz

    result = search_csv(file_path, name, surname, age)  # CSV faylda ma'lumotni qidiramiz

    if result:
        print("Malumotlar bazasidan ma'lumot topildi: ", result)  # Ma'lumot topilsa uni chiqaramiz
    else:
        print("Bunday ma'lumot topilmadi.")  # Ma'lumot topilmagan holatda xabarni chiqaramiz
        print("Iltimos registratsiyadan o'ting")  # Registratsiyadan o'tishni tavsiya qilamiz
        Registratsiya()  # Registratsiyadan o'tish funksiyasini chaqirish

def Registratsiya():
    cam = cv2.VideoCapture(1)  # Kamera obyektini yaratamiz
    cam.set(3, 640)  # Kamera o'lchamlarini sozlaymiz
    cam.set(4, 480)

    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Yuz tanishlashingizni aniqlash uchun kaskad klasifikatorini yaratamiz

    face_id = random.randint(1, 999)  # Foydalanuvchidan ID raqamini so'raymiz
    person_name = input("Ismingizni kiriting: ")  # Foydalanuvchidan ismni so'raymiz
    person_surname = input("Familyangizni kiriting: ")  # Foydalanuvchidan familiyani so'raymiz
    age = input("Yoshingizni Kiriting: ")  # Foydalanuvchidan yoshni so'raymiz

    with open("user_info.csv", 'a', newline='\n') as file:
        csv_writer = csv.writer(file)  # CSV faylga ma'lumot yozish obyektini yaratamiz
        csv_writer.writerow([person_name, person_surname, age, face_id])  # Ma'lumotlarni CSV faylga yozamiz

    print("Suratlarni olish, kamera tomonga qarang ....... ")  # Foydalanuvchiga suratlar olinayotganini aytamiz
    count = 0  # Suratlar sonini hisoblash uchun o'zgaruvchi

    while True:
        ret, img = cam.read()  # Kameradan surat olish
        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Rangli suratni oq rangga o'tkazish

        faces = detector.detectMultiScale(converted_image, 1.3, 5)  # Yuzlarni aniqlash
        if len(faces) > 0:
           for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Yuzga bo'lgan qirqib chiqish chizigini joylashtiramiz
            count += 1

            cv2.imwrite("samples/face." + str(face_id) + '.' + str(count) + ".jpg", converted_image[y:y + h, x:x + w])  # Suratni samples papkasiga saqlash
            cv2.imshow('image', img)  # Suratni namoyish etish

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 250:
            count = 0
            break

    print("Suratlar muvaffaqiyatli olingan!")  # Suratlar muvaffaqiyatli olinishini aytamiz
    from Model_Trainer import train_face_recognition_model, samples_path, cascade_file, output_model_file
    train_face_recognition_model(samples_path, cascade_file, output_model_file)
    cam.release()  # Kamera obyektini qisqartirish
    cv2.destroyAllWindows()  # Barcha OpenCV oynalarni yopish

def Imtihonga_kirish():
    sorovnoma()  # Foydalanuvchining ma'lumotlarini olish
    cam = cv2.VideoCapture(1)  # Kamera obyektini yaratamiz
    cam.set(3, 1266)  # Kamera o'lchamlarini sozlaymiz
    cam.set(4, 768)

    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Yuz tanishlashingizni aniqlash uchun kaskad klasifikatorini yaratamiz
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Yuzlarni tanish uchun model yaratamiz
    recognizer.read('trainer/trainer.yml')  # Modelni o'qish

    font = cv2.FONT_HERSHEY_SIMPLEX  # Matn shrifti

    minW = 0.1 * cam.get(3)  # Minimal yuz o'lchami
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()  # Kameradan surat olish
        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Rangli suratni oq rangga o'tkazish

        faces = detector.detectMultiScale(converted_image, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))  # Yuzlarni aniqlash

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Yuzga bo'lgan qirqib chiqish chizigini joylashtiramiz
                id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])  # Yuzni tanish
                accuracy = 100 - int(accuracy)

                if accuracy >= 60:
                    cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)  # ID raqamini chiqaramiz
                    cv2.putText(img, "Imtihonga kiring! O'xshashlik: " + str(accuracy) + "%", (x + 5, y - 55), font, 1, (0, 255, 0), 2)  # Imtihon uchun taklif
                else:
                    cv2.putText(img, "Ro'hatdan o'ting! O'xshashlik: " + str(accuracy) + "%", (x + 5, y - 55), font, 1, (0, 0, 255), 2)  # Ro'yxatdan o'tish uchun taklif

        cv2.imshow('camera', img)  # Suratni namoyish etish
        k = cv2.waitKey(10) & 0xff

        if k == 27:
            break

    print("Dasturdan foydalanish uchun rahmat, hayr!")  # Dasturdan foydalanish uchun rahmat
    cam.release()  # Kamera obyektini qisqartirish
    cv2.destroyAllWindows()  # Barcha OpenCV oynalarni yopish

def main():
    while True:
        print("1. Imtihonga kirish")
        print("2. Registratsiya")
        print("3. Chiqish")
        choice = input("Tanlang: ")  # Foydalanuvchidan tanlovni so'raymiz

        if choice == "1":
            Imtihonga_kirish()  # Imtihon tekshirish funksiyasini chaqirish
        elif choice == "2":
            Registratsiya()  # Imtihon qabul qilish funksiyasini chaqirish
        elif choice == "3":
            break
        else:
            print("Noto'g'ri tanlov! Iltimos, qayta kiriting.")  # Noto'g'ri tanlovda xabarni chiqaramiz

if __name__ == '__main__':
    main()  # Asosiy funksiyani chaqirish
