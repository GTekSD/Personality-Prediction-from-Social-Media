## Personality Detector Page ##

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 5 03:30:41 2021

@author: Сухас Дхолз
"""

import pandas as pd, numpy as np, re
from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn import svm
from PIL import Image, ImageTk
from tkinter import ttk
from joblib import dump , load
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
import nltk

nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
  
    
root = tk.Tk()
root.title("Personality Prediction using Twitter")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

# image2 =Image.open(r'C:/Users/GTekSD/Desktop/personality_Prediction/BG.jpg')

# image2 =image2.resize((w,h), Image.ANTIALIAS)

# background_image=ImageTk.PhotoImage(image2)

# background_label = tk.Label(root, image=background_image)

# background_label.image = background_image

# background_label.place(x=0, y=0)

img=ImageTk.PhotoImage(Image.open("s1.jpg"))

img2=ImageTk.PhotoImage(Image.open("s2.jpg"))

img3=ImageTk.PhotoImage(Image.open("s3.jpg"))


logo_label=tk.Label()
logo_label.place(x=0,y=100)

x = 1

# function to change to next image
def move():
	global x
	if x == 4:
		x = 1
	if x == 1:
		logo_label.config(image=img)
	elif x == 2:
		logo_label.config(image=img2)
	elif x == 3:
		logo_label.config(image=img3)
	x = x+1
	root.after(4000, move)

# calling the function
move()

w = tk.Label(root, text="Personality Prediction",width=40,background="#7D0552",height=2,font=("Times new roman",28,"bold"))
w.place(x=500,y=10)

w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.configure(background="#7D0552")



def Train():
    
    result = pd.read_csv(r"C:/Users/GTekSD/Desktop/personality_Prediction/new.csv",encoding = 'unicode_escape')

    result.head()
        
    result['posts'] = result['posts'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    
    def pos(review_without_stopwords):
        return TextBlob(review_without_stopwords).tags
    
    
    os = result.posts.apply(pos)
    os1 = pd.DataFrame(os)
    
    os1.head()
    
    os1['pos'] = os1['posts'].map(lambda x: " ".join(["/".join(x) for x in x]))
    
    result = result = pd.merge(result, os1, right_index=True, left_index=True)
    result.head()
    result['pos']
    review_train, review_test, label_train, label_test = train_test_split(result['pos'], result['type'],
    test_size=0.2, random_state=13)
    
    tf_vect = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False)
    
    X_train_tf = tf_vect.fit_transform(review_train)
    X_test_tf = tf_vect.transform(review_test)
    
    
    def svc_param_selection(X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        return grid_search.best_params_
    
    
    svc_param_selection(X_train_tf, label_train, 2)
    
    clf = svm.SVC(C=10, gamma=0.001, kernel='linear')   
    clf.fit(X_train_tf, label_train)
    pred = clf.predict(X_test_tf)
    
    with open('vectorizer.pickle', 'wb') as fin:
        pickle.dump(tf_vect, fin)
	
    with open('mlmodel.pickle', 'wb') as f:
        pickle.dump(clf, f)
    
    pkl = open('mlmodel.pickle', 'rb')
    clf = pickle.load(pkl)
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    
    X_test_tf = tf_vect.transform(review_test)
    pred = clf.predict(X_test_tf)
    
    print(metrics.accuracy_score(label_test, pred))
	
    print(confusion_matrix(label_test, pred))

    print(classification_report(label_test, pred))
       
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(label_test, pred)))
    print("Accuracy : ",accuracy_score(label_test, pred)*100)
    accuracy = accuracy_score(label_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    ACC = (accuracy_score(label_test, pred) * 100)
    repo = (classification_report(label_test, pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as SVM_MODEL.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=320)
    
    dump (clf,"SVM_MODEL.joblib")
    print("Model saved as SVM_MODEL.joblib")

frame = tk.LabelFrame(root,text="Control Panel",width=600,height=400,bd=3,background="#F67280",font=("Tempus Sanc ITC",15,"bold"))
frame.place(x=600,y=200)

entry = tk.Entry(frame,width=50)
entry.insert(0,"Enter text here...")
entry.place(x=25,y=150)

def Test():
    predictor = load("SVM_MODEL.joblib")
    Given_text = entry.get()
    #Given_text = "the 'roseanne' revival catches up to our thorny po..."
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    X_test_tf = tf_vect.transform([Given_text])
    y_predict = predictor.predict(X_test_tf)
    print(y_predict[0])
	
    if y_predict[0]==0:
        label4 = tk.Label(root,text ="Personality Prediction is Introvrsion,Intuition,Feeling,Judging",width=70,height=2,bg='Green',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==1:
        label4 = tk.Label(root,text ="Personality Prediction is Extroversion,Intuition,Thinking,Perceiving",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==2:
        label4 = tk.Label(root,text ="Personality Prediction is Introvrsion,Intuition,Thinking,Perceiving",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==3:
        label4 = tk.Label(root,text ="Personality Prediction is Introvrsion,Intuition,Thinking,Judging",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==4:
        label4 = tk.Label(root,text ="Personality Prediction is Extroversion,Intuition,Thinking,Judging",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==5:
        label4 = tk.Label(root,text ="Personality Prediction is Extroversion,Intuition,Feeling,Judging",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==6:
        label4 = tk.Label(root,text ="Personality Prediction is Introvrsion,Intuition,Feeling,Perceiving",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==7:
        label4 = tk.Label(root,text ="Personality Prediction is Extroversion,Intuition,Feeling,Perceiving",width=70,eight=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==8:
        label4 = tk.Label(root,text ="Personality Prediction is Extroversion,Intuition,Feeling,Perceiving",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==9:
        label4 = tk.Label(root,text ="Personality Prediction is Introvrsion,Sensing,Thinking,Perceiving",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==10:
        label4 = tk.Label(root,text ="Personality Prediction is Introvrsion,Sensing,Feeling,Judging",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==11:
        label4 = tk.Label(root,text ="Personality Prediction is Introvrsion,Sensing,Thinking,Judging",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==12:
        label4 = tk.Label(root,text ="Personality Prediction is Extroversion,Sensing,Thinking,Perceiving",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==13:
        label4 = tk.Label(root,text ="Personality Prediction is Introvrsion,Intuition,Thinking,Judging",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==14:
        label4 = tk.Label(root,text ="Personality Prediction is Introvrsion,Intuition,Thinking,Judging",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
	
    elif y_predict[0]==15:
        label4 = tk.Label(root,text ="Personality Prediction is Introvrsion,Intuition,Thinking,Judging",width=70,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=600,y=800)
         
    
def window():
    root.destroy()    
    


# = tk.Button(frame,command=Train,text="Train",bg="red",fg="black",width=15,font=("Times New Roman",15,"italic"))
#button2.place(x=5,y=100)

button3 = tk.Button(frame,command=Test,text="Test",bg="green",fg="black",width=15,font=("Times New Roman",20,"italic"))
button3.place(x=50,y=250)

exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=800, y=650)


root.mainloop()
