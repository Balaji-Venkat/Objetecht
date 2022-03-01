import tkinter as tk
import schedule
import cv2
import numpy as np
import time
import pyttsx3
import random
import speech_recognition as sr

from tkinter import font
from PIL import ImageTk, Image
from deepface import DeepFace as df
from random import randint
from playsound import playsound
from textblob import TextBlob as tb
from pygame import mixer

title_font = ("Comic Sans MS", 50, "italic")
font = ("Comic Sans MS", 10, "normal")
w = 800
h = 200
config = "C:/Users/Bala/Downloads/yolov3-spp.cfg.txt"
weight = "C:/Users/Bala/Downloads/yolov3-spp.weights"
conf_Threshold = 0.5
nms_Threshold = 0.3
a = (320,320)
font = cv2.FONT_HERSHEY_SIMPLEX
fire_cascade = cv2.CascadeClassifier("fire_detection.xml")
scale = 1.1
confidence = 9
fontSize = 0.3
fontThickness = 1
contourThickness = 2
i = 0
colours = ['black','brown','gray','blue','green','indigo','purple','white','pink','yellow','orange','magenta','red','lime','gold']
bgcolours = ['black','brown','gray','blue','green','indigo','purple']
fgcolours = ['white','pink','yellow','orange','magenta','red','lime','gold']

def cursor_hover(button, color1, color2):
    button.bind("<Enter>", func=lambda e: button.config(foreground = color1, background=color2))
    button.bind("<Leave>", func=lambda e: button.config(background=color1, foreground = color2))
    
def findObjects(outputs,img):
    hT, wT, cT = img.shape
    box = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]* wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                box.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(box, confs,confThreshold, nmsThreshold)

    for i in indices:
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img1, (x,y),(x+w,y+h),(255,0,255),2)
        global label
        label = str(classes[classId[i]])
        cv2.putText(img1, label, (x, y), font, fontSize, (r, g, b), fontThickness)
        results2 = tk.Label(r2, text='I can detect a ' + label)
        c2.create_window(100*i, 400, window=results2)
        
def get_connotation(sent):
    analysis = tb(sent)
    value = analysis.sentiment.polarity
    if value > 0:
        return ('positive')
    elif value == 0:
        return ('neutral')
    else:
        return ('negative')
        
def response():
    text = str(entry.get())
    records_all = text.split('.')
    c = 1
    for sentence in records_all:
        connotation = str(get_connotation(sentence))
        MyText = sentence + ' has a ' + connotation + ' connotation'
        connotation1 = tk.Label(r3, text=MyText)
        c3.create_window(200*c, 300, window=connotation1)
        c = c+1
        SpeakText(MyText)
        
def SpeakText(command):
    x = pyttsx3.init()
    x.say(command) 
    x.runAndWait()

def recognize_speech(recognizer, microphone):
    speech = {}
    label1 = tk.Label(r4, text='Please speak now')
    c4.create_window(400, 250, window=label1)
    with microphone as source:
        try:
            audio = recognizer.listen(source)
            speech= recognizer.recognize_google(audio)
            connotation = get_connotation(speech)
            MyText = speech + ' has a ' + connotation + ' connotation'
            connotation2 = tk.Label(r4, text=MyText)
            c4.create_window(400, 250, window=connotation2)
            SpeakText(MyText)
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            print("Unknown value was entered")
            

    return speech
    label2 = tk.Label(r4, text=' '*100)
    c4.create_window(400, 250, window=label2)

def respond(color):
    x = int(rating.get())
    global response1
    global response2
    global response3
    if x > 0 and x < 3:
        response1 = tk.Label(r6, text='     Please tell us how to improve      ', fg=random.choice(fgcolours), bg=color)
        response2 = tk.Label(r6, text='Thank you for the valuable feedback', fg=random.choice(fgcolours), bg=color)
        response3 = tk.Label(r6, text=' '*75, fg=random.choice(fgcolours), bg=color)
        x = 'Please tell us how to improve'
        y = 'Thank you for the valuable feedback'
    elif x == 3:
        response1 = tk.Label(r6, text='Thank you for the valuable feedback', fg=random.choice(fgcolours), bg=color)
        response2 = tk.Label(r6, text='We will continue to improve the app', fg=random.choice(fgcolours), bg=color)
        response3 = tk.Label(r6, text=' '*75, fg=random.choice(fgcolours), bg=color)
        x = 'Thank you for the valuable feedback'
        y = 'We will continue to improve the app'
    elif x > 3 and x < 6:
        response1 = tk.Label(r6, text=' '*75, fg=random.choice(fgcolours), bg=color)
        response2 = tk.Label(r6, text=' '*75, fg=random.choice(fgcolours), bg=color)
        response3 = tk.Label(r6, text='Thank you for the valuable feedback', fg=random.choice(fgcolours), bg=color)
        x = ''
        y = 'Thank you for the valuable feedback'
    else:
        response1 = tk.Label(r6, text='         Please enter a number from 1-5       ', fg=random.choice(fgcolours), bg=color)
        response2 = tk.Label(r6, text=' '*75, fg=random.choice(fgcolours), bg=color)
        response3 = tk.Label(r6, text=' '*75, fg=random.choice(fgcolours), bg=color)
        x = 'Please enter a number from 1-5'
        y = ''
        
    c6.create_window(400, 300, window=response1)
    c6.create_window(400, 350, window=response2)
    c6.create_window(400, 325, window=response3)
    SpeakText(x)
    SpeakText(y)


def screen1(rx): #Supposed to switch to canvas 1
    rx.destroy()
    canvas1()
    
def screen2(): #Supposed to switch to canvas 2
    r1.destroy()  
    canvas2()
    
def screen3(): #Supposed to switch to canvas 3
    r1.destroy()  
    canvas3()

def screen4(): #Supposed to switch to canvas 4
    r1.destroy()  
    canvas4()
    
def screen5(): #Supposed to switch to canvas 5
    r1.destroy()  
    canvas5()
    
def screen6(): #Supposed to switch to canvas 6
    r1.destroy()  
    canvas6()
    

def canvas0():
    global r0
    r0 = tk.Tk()
    r0.title('Objtecht')
    c0 = tk.Canvas(r0, width = 800, height = 500, bg=random.choice(colours)) #Pre-page
    c0.pack()

    icon = tk.PhotoImage( file = r'Logo.png')
    title = tk.Label(r0, text='Objetect', fg=random.choice(fgcolours), bg=random.choice(bgcolours))
    title.configure(font = ("Comic Sans MS", 50, "italic"))
    c0.create_window(400, 150, window=title)
    r0.iconphoto(True, icon)
    image = Image.open("Logo.png")
    logo = ImageTk.PhotoImage(image)
    Logo = tk.Label(image=logo)
    Logo.image = logo
    Logo.place(x=320, y=250)
    r0.after(5000, r0.destroy)
    r0.mainloop()

def canvas1(): #If canvas 1 is open do the following:
    global r1
    r1 = tk.Tk()
    r1.title('Home')
    c1 = tk.Canvas(r1, width = 800, height = 500, bg=random.choice(colours))
    c1.pack()
        
    welcome = tk.Label(r1, text='Welcome To Objetecht ', fg=random.choice(fgcolours), bg=random.choice(bgcolours))
    welcome.configure(font = ("Comic Sans MS", 30, "normal"))
    c1.create_window(400, 100, window=welcome)
    description = tk.Label(r1, text='Detect everthing from a teddy bear to fire with tech', fg=random.choice(fgcolours), bg=random.choice(bgcolours))
    welcome.configure(font = ("Comic Sans MS", 20, "normal"))
    c1.create_window(400, 200, window=description)

    opt2 = tk.Button(r1, text='Camtecht', fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = screen2)
    cursor_hover(opt2, random.choice(bgcolours), random.choice(fgcolours))
    c1.create_window(300, 300, window=opt2)
    opt3 = tk.Button(r1, text='Textecht', fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = screen3)
    cursor_hover(opt3, random.choice(bgcolours), random.choice(fgcolours))
    c1.create_window(400, 300, window=opt3)
    opt4 = tk.Button(r1, text='Speetecht', fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = screen4)
    cursor_hover(opt4, random.choice(bgcolours), random.choice(fgcolours))
    c1.create_window(500, 300, window=opt4)
    opt5 = tk.Button(r1, text='Help', fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = screen5)
    cursor_hover(opt5, random.choice(bgcolours), random.choice(fgcolours))
    c1.create_window(350, 400, window=opt5)
    opt6 = tk.Button(r1, text='Rate Us', fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = screen6)
    cursor_hover(opt6, random.choice(bgcolours), random.choice(fgcolours))
    c1.create_window(450, 400, window=opt6)
    r1.mainloop()


import cv2
from playsound import playsound
fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

def firedetection():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fire = fire_cascade.detectMultiScale(frame, 1.2, 5)

        for (x,y,w,h) in fire:
            cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.imshow('Fire Detection', frame)
            SpeakText("Fire is detected")
            mixer.init() 
            mixer.music.load('alarm.mp3') 
            mixer.music.play() 
            time.sleep(5)
            mixer.music.stop()
            
        cv2.imshow('Fire Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            cv2.destroyWindow('Fire Detection')
        
def shapesdetection():
    while True:
        cap = cv2.VideoCapture(0)
        frame, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        i = 0
        

        for contour in contours:
            if i == 0:
                i = 1
                continue

            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

            if len(approx) == 3:
                cv2.putText(img, 'Triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif len(approx) == 4:
                cv2.putText(img, 'Quadrilateral', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif len(approx) == 5:
                cv2.putText(img, 'Pentagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif len(approx) == 6:
                cv2.putText(img, 'Hexagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(img, 'circle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Shapes',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            cv2.destroyWindow("Shapes")

def objectdetection():
    net = cv2.dnn.readNet(weight, config)
    classes = []
    with open("C:/Users/Bala/Downloads/coco.names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayersNames()

    while True:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened:
            raise IOError('Cannot open webcam')

        success, img = cap.read()

        blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0,), True, crop=False)

        net.setInput(blob)
        outp = net.forward(output_layers)

        height, width, channels = img.shape
        class_ids = []
        confidences = []
        boxes = []

        for out in outp:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.8:

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_Threshold, nms_Threshold)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            r = randint(0,255)
            g = randint(0,255)
            b = randint(0,255)
            cv2.rectangle(img, (x, y), (x + w, y + h), (r,g,b), 2)
            cv2.putText(img, label, (x, y), font, 1, (r,g,b), 3, 1)
            SpeakText('I can detect a ' + label)

        cv2.imshow('Image',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            cv2.destroyWindow("Image")
            
def canvas2(): #If canvas 2 is open do the following:
    global r2
    global c2
    r2 = tk.Tk()
    r2.title('Camtecht')
    c2 = tk.Canvas(r2, width = 800, height = 500, bg=random.choice(colours))
    c2.pack()
    opt2 = tk.Button(r2, text='Fire Detection',fg=random.choice(fgcolours), bg=random.choice(bgcolours), command =firedetection)
    cursor_hover(opt2, random.choice(bgcolours), random.choice(fgcolours))
    c2.create_window(200, 200, window=opt2)
    opt3 = tk.Button(r2, text='Shapes Detection',fg=random.choice(fgcolours), bg=random.choice(bgcolours), command =shapesdetection)
    cursor_hover(opt3, random.choice(bgcolours), random.choice(fgcolours))
    c2.create_window(400, 200, window=opt3)
    opt4 = tk.Button(r2, text='Object detection',fg=random.choice(fgcolours), bg=random.choice(bgcolours), command =objectdetection)
    cursor_hover(opt4, random.choice(bgcolours), random.choice(fgcolours))
    c2.create_window(600, 200, window=opt4)
    label1 = tk.Label(r2, text='Press q, then the close button to close the opened window', fg=random.choice(fgcolours), bg=random.choice(bgcolours))
    c2.create_window(400, 300, window=label1)
    opt1 = tk.Button(r2, text='<--Home',fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = lambda:screen1(r2))
    cursor_hover(opt1, random.choice(bgcolours), random.choice(fgcolours))
    c2.create_window(400, 400, window=opt1)

def canvas3(): #If canvas 3 is open do the following:
    global r3
    global c3
    r3 = tk.Tk()
    r3.title('Textecht')
    c3 = tk.Canvas(r3, width = 800, height = 500, bg=random.choice(colours))
    c3.pack()
    
    label = tk.Label(r3, text='Please do not include any punctuation other than periods',fg=random.choice(fgcolours), bg=random.choice(bgcolours))
    c3.create_window(400, 200, window=label)
    global entry
    entry = tk.Entry(r3)
    c3.create_window(400, 250, window=entry)
    submit = tk.Button(r3, text='Submit',fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = response)
    cursor_hover(submit, random.choice(bgcolours), random.choice(fgcolours))
    c3.create_window(400, 350, window=submit)
        
    opt1 = tk.Button(r3, text='<--Home',fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = lambda:screen1(r3))
    cursor_hover(opt1, random.choice(bgcolours), random.choice(fgcolours))
    c3.create_window(400, 400, window=opt1)

def speechdetection():
    recog = sr.Recognizer()
    mic = sr.Microphone(device_index=1)   
    recognize_speech(recog, mic)

def canvas4(): #If canvas 4 is open do the following:
    global r4
    global c4
    r4 = tk.Tk()
    r4.title('Speetecht')
    c4 = tk.Canvas(r4, width = 800, height = 500, bg=random.choice(colours))
    c4.pack()
    
    opt1 = tk.Button(r4, text='Detect Speech',fg=random.choice(fgcolours), bg=random.choice(bgcolours), command =speechdetection)
    cursor_hover(opt1, random.choice(bgcolours), random.choice(fgcolours))
    c4.create_window(400, 200, window=opt1)

    opt2 = tk.Button(r4, text='<--Home',fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = lambda:screen1(r4))
    cursor_hover(opt2, random.choice(bgcolours), random.choice(fgcolours))
    c4.create_window(400, 300, window=opt2)
    r4.mainloop()
        
def canvas5(): #If canvas 5 is open do the following:
    global r5
    r5 = tk.Tk()
    r5.title('Help')
    c5 = tk.Canvas(r5, width = 800, height = 500, bg=random.choice(colours))
    c5.pack()
    
    label1 = tk.Label(r5, text ="Camtecht: Detects fire, shapes, and objects \nusing the camera or an uploaded image.", fg=random.choice(fgcolours), bg=random.choice(bgcolours), justify='left')
    c5.create_window(400,100, window = label1)
    label2 = tk.Label(r5, text ="Textecht: Detects emotion of the text entered.", fg=random.choice(fgcolours), bg=random.choice(bgcolours))
    c5.create_window(400,200, window = label2)
    label3 = tk.Label(r5, text ="Speetecht: Detects connotation of the words spoken.", fg=random.choice(fgcolours), bg=random.choice(bgcolours))
    c5.create_window(400,300, window = label3)
    
    opt1 = tk.Button(r5, text='<--Home',fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = lambda:screen1(r5))
    cursor_hover(opt1, random.choice(bgcolours), random.choice(fgcolours))
    c5.create_window(400, 400, window=opt1)

def canvas6(): #If canvas 6 is open do the following:
    global r6
    global c6
    r6 = tk.Tk()
    r6.title('Rate Us')
    bgc = random.choice(colours)
    c6 = tk.Canvas(r6, width = 800, height = 500, bg=bgc)
    c6.pack()
    
    label = tk.Label(r6, text='Please rate us from 1-5', fg=random.choice(fgcolours), bg=random.choice(bgcolours))
    c6.create_window(400, 150, window=label)
    global rating
    rating = tk.Entry(r6)
    c6.create_window(400, 200, window=rating)
    submit = tk.Button(r6, text='Submit',fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = lambda:respond(bgc))
    cursor_hover(submit, random.choice(bgcolours), random.choice(fgcolours))
    c6.create_window(400, 250, window=submit)
    
    opt1 = tk.Button(r6, text='<--Home', fg=random.choice(fgcolours), bg=random.choice(bgcolours), command = lambda:screen1(r6))
    cursor_hover(opt1, random.choice(bgcolours), random.choice(fgcolours))
    c6.create_window(400, 400, window=opt1)
    
canvas0()
r0.after(5000, canvas1())
r1.mainloop()
