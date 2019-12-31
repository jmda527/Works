import tkinter as tk
from model_create import Trash_sep
import pickle

def predict():
    var = c.get()
    var = seperator.predicting(var)
    a.config(text=var)

def modify(): m
    seperator = Trash_sep()
    seperator.preproccessing()
    with open('trash_sep.pkl','wb') as f:
        pickle.dump(seperator,f,pickle.HIGHEST_PROTOCOL)
    with open('trash_sep.pkl','rb') as f:
        seperator = pickle.load(f)

try:
    with open('trash_sep.pkl','rb') as f:
        seperator = pickle.load(f)
except:
    modify()
    with open('trash_sep.pkl','rb') as f:
        seperator = pickle.load(f)

window = tk.Tk()
window.title('公文分類器')
window.geometry('500x200')

tk.Label(window, text='請輸入關鍵字: ').place(x=30, y= 100)
a = tk.Label(window,text='請輸入關鍵字...',bg='yellow',width=72,height=2)
b = tk.Button(window,text='分類',bg='blue',fg='white',width=10,height=2,command=predict)
c = tk.Entry(window,width=50)
d = tk.Button(window,text='更新',bg='red',fg='white',width=5,height=2,command=modify)

a.place(x=0,y=10)
b.pack(side='bottom')
c.place(x=120,y=100)
d.place(x=455,y=159)
window.mainloop()