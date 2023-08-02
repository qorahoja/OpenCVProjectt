import cv2
import numpy as np
import csv
import pandas as pd
import random
from tkinter import Tk, Label, Entry, Button

def search_csv(file_path, name, surname, age):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) >= 3 and row[0] == name and row[1] == surname and int(row[2]) == age:
                return row
    return None


face_id = random.randint(1, 999)

def hide(*widgets):
    for widget in widgets:
        widget.pack_forget()
def face_idd():
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
def sorovnoma():
    def handle_search():
        name = name_entry.get()
        surname = surname_entry.get()
        age = int(age_entry.get())
        result = search_csv("user_info.csv", name, surname, age)
        if result:
            result_label.config(text="Ma'lumotlar bazasidan ma'lumot topildi: " + str(result))
            face_idd()
        else:
            result_label.config(text="Bunday ma'lumot topilmadi.\nIltimos, registratsiyadan o'ting.")
            Registratsiya()
        # Imtihonga_kirish()

    window = Tk()
    window.title("Ma'lumot qidirish")
    window.geometry("800x600")
    window.configure(bg='blue')

    name_label = Label(window, text="Ismingizni kiriting:")
    name_label.pack()
    name_entry = Entry(window)
    name_entry.pack()

    surname_label = Label(window, text="Familyangizni kiriting:")
    surname_label.pack()
    surname_entry = Entry(window)
    surname_entry.pack()

    age_label = Label(window, text="Yoshingizni kiriting:")
    age_label.pack()
    age_entry = Entry(window)
    age_entry.pack()

    search_button = Button(window, text="Qidirish", command=handle_search, activebackground='black', activeforeground='White')
    search_button.pack()

    result_label = Label(window, text="")
    result_label.pack()

    # window.mainloop()

def surat_olish():
    cam = cv2.VideoCapture(1)  # Kamera obyektini yaratamiz
    cam.set(3, 640)  # Kamera o'lchamlarini sozlaymiz
    cam.set(4, 480)

    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Yuz tanishlashingizni aniqlash uchun kaskad klasifikatorini yaratamiz
    
    
   

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



def Registratsiya():
    def handle_registration():
        
        person_name = name_entry.get()
        person_surname = surname_entry.get()
        age = age_entry.get()
        with open("user_info.csv", 'a', newline='\n') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([person_name, person_surname, age, face_id])
        result_label.config(text="Registratsiya muvaffaqiyatli yakunlandi!")
        surat_olish()

    window = Tk()
    window.title("Registratsiya")
    window.geometry("800x600")
    window.configure(bg='blue')
    

    name_label = Label(window, text="Ismingizni kiriting:")
    name_label.pack()
    name_entry = Entry(window)
    name_entry.pack()

    surname_label = Label(window, text="Familyangizni kiriting:")
    surname_label.pack()
    surname_entry = Entry(window)
    surname_entry.pack()

    age_label = Label(window, text="Yoshingizni kiriting:")
    age_label.pack()
    age_entry = Entry(window)
    age_entry.pack()

    registration_button = Button(window, text="Registratsiya", command=handle_registration, activebackground='black', activeforeground='White')
    registration_button.pack()

    result_label = Label(window, text="")
    result_label.pack()

        
    window.mainloop()
    

def Imtihonga_kirish():
    
    sorovnoma()



def handle_exit():
        window = Tk()
        window.destroy()  

def main():
    
        
    

    window = Tk()
    window.title("Asosiy menyu")
    window.geometry("800x600")
    window.configure(bg='blue')
    
  
    exam_button = Button(window, text="Imtihonga kirish", command=Imtihonga_kirish, activebackground='black', activeforeground='White', width=40, height=4)
    exam_button.pack()

    exam_button.place(x=265, y=120)

    registration_button = Button(window, text="Registratsiya", command=Registratsiya, activebackground='black', activeforeground='White', width=40, height=4)
    registration_button.pack()
    registration_button.place(x=265, y=275)
    exit_button = Button(window, text="Chiqish", command=quit,activebackground='black', activeforeground='white', width=40, height=4)
    exit_button.pack()
    exit_button.place(x=265, y=430)
   

    window.mainloop()

main()