import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model
from os import path

model = load_model("age_gender.h5")

gender_dict = {0: 'Male', 1: "Female"}

cap = None  # Biến để lưu trữ đối tượng VideoCapture
camera_opened = False  # Biến kiểm tra xem camera đã được mở hay chưa

root = tk.Tk()
root.title("ƯỚC TÍNH GIỚI TÍNH VÀ TUỔI")
root.geometry("1400x700") 

text6 = tk.Label(root, text="", foreground="red", font=("Arial", 20))
text6.place(x = 875, y = 525 )

text7 = tk.Label(root, text="", foreground="red", font=("Arial", 20))
text7.place(x = 875, y = 575 )


# Tạo đối tượng CascadeClassifier để phát hiện khuôn mặt trong ảnh
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')


#-------------------HÀM ƯỚC TÍNH TUỔI VÀ GIỚI TÍNH TỪ ẢNH LẤY TỪ CAMERA-----------------------

def show_image():
    global cap, camera_opened  

    if cap is not None: #ktra camera còn bật không
        cap.release() 
        camera_opened = False

    filename = entry.get()

    if not filename:  #kiểm tra đã nhập vào entry chưa

        text7.config(text="CHUA NHAP FILE", foreground="red", font=("Arial", 20))
        text6.config(text="", foreground="red", font=("Arial", 20))

    else:
        image_link = path.join('E:\MV_Python\MV_Python\ANH_AI', filename)
        temp = filename.split(' ')
        age_org = int(temp[0])
        gender_org= int(temp[1])

        text_org = "Original Age: {} years, Gender: {}  ".format(age_org, gender_dict[gender_org]) 
        text6.config(text=text_org, foreground="red", font=("Arial", 20))

        #Xử lý ảnh để phù hợp với đầu vào cảu model
        image = Image.open(image_link)
        face_img = np.array(image)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)             
        face_img = face_img.astype('float32') / 255.0
        face_img = cv2.resize(face_img, (128, 128))
        face_img = np.expand_dims(face_img, axis=0)
        
        #Dự đoán tuổi và giới tính
        pred = model.predict(face_img)

        age = round(pred[1][0][0])
        gender = gender_dict[round(pred[0][0][0])]
        
        text_predict = "Predict Age: {} years, Gender: {}  ".format(age, gender)
        text7.config(text=text_predict, foreground="red", font=("Arial", 20))
        
        photo = ImageTk.PhotoImage(image)

        label_image.config(image=photo)
        label_image.image = photo


#-------------------HÀM ƯỚC TÍNH TUỔI VÀ GIỚI TÍNH TỪ ẢNH LẤY TỪ CAMERA-----------------------

def open_camera():
    global cap, camera_opened 
    text6.config(text="", foreground="red", font=("Arial", 20))
    text7.config(text="", foreground="red", font=("Arial", 20))

    if not camera_opened:
        cap = cv2.VideoCapture(0)  
        camera_opened = True
   
    def update_frame():
        ret, frame = cap.read()  

        if ret:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(frame_rgb, 1.3, 5)

            for (x,y,w,h) in faces:
                face_img = frame[y-10:y+h, x-5:x+w]
                
                if face_img is not None:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    face_img = face_img.astype('float32') / 255.0
                    face_img = cv2.resize(face_img, (128, 128))
                    face_img = np.expand_dims(face_img, axis=0)
    
                    pred = model.predict(face_img)

                    age = round(pred[1][0][0])
                    gender = gender_dict[round(pred[0][0][0])]
                    
                    text = "Predict Age: {} years, Gender: {}".format(age, gender)
                    cv2.putText(frame_rgb, text, (x-100, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.rectangle(frame_rgb,(x,y-25),(x+w,y+h),(255,0,0),2)
            
            
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)

            label_image.config(image=photo)
            label_image.image = photo

        #if camera_opened:
        #    label_image.after(1000, update_frame)

    update_frame()

#---------------------------------------------------------------------------



button_show_image = tk.Button(root, text="Lấy ảnh từ file", font=("Arial", 20), command=show_image, width=15, height=2)
button_show_image.place(x=545, y=50) 


button_open_camera = tk.Button(root, text="Lấy ảnh từ camera", font=("Arial", 20), command=open_camera, width=15, height=2)
button_open_camera.place(x=545, y=200) 

entry = tk.Entry(root)
entry.place(x = 550, y = 350)
entry.config(width=30)

label_image = tk.Label(root)
label_image.config(width=500, height=500)
label_image.place(x=800, y=25)

text1 = tk.Label(root, text="MÔN TRÍ TUỆ NHÂN TẠO", foreground="red", font=("Arial", 20))
text1.place(x = 75, y = 150 )

text2 = tk.Label(root, text="BÁO CÁO CUỐI KỲ", foreground="black", font=("Arial", 16))
text2.place(x = 135, y = 200 )

text3 = tk.Label(root, text="HỆ THỐNG DỰ ĐOÁN GIỚI TÍNH VÀ TUỔI", foreground="black", font=("Arial", 16))
text3.place(x = 50, y = 250 )

text4 = tk.Label(root, text="SVTH: Quãng Trọng Tấn Lộc 20146194", foreground="black", font=("Arial", 16))
text4.place(x = 150, y = 300 )

text5 = tk.Label(root, text="GVHD: PGS.TS Nguyễn Trường Thịnh", foreground="black", font=("Arial", 16))
text5.place(x = 150, y = 350 )


# Hiển thị giao diện chính
root.mainloop()
