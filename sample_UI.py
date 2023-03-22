import tkinter as tk   
from tkinter import filedialog


def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    video_path = filename
    print('Selected:', filename)   

root = tk.Tk()
button = tk.Button(root, text='Upload a Video', command=UploadAction)
button.pack()

L1 = tk.Label(text="Input questions: ", font = ("Comic Sans MS", 20, "bold"))
L1.pack()
textField = tk.Entry(root, width = 100)
textField.pack(fill=tk.NONE)
ask_button = tk.Button(root, text='Ask', command=UploadAction)
ask_button.pack()

question = ['What color is her dress?',
            'Where is she sitting on?',
            'How many people are there?',
            'What is she facing to?',
            'Is it a sunny day?',
           'What is above the ocean?']

answer = ['pink','beach','1','ocean','yes','cloud']


from PIL import Image, ImageTk

# Create a photoimage object of the image in the path
image1 = Image.open("sample_pic.png")
test = ImageTk.PhotoImage(image1)

label0 = tk.Label(text='The sample frame:', font = ("Comic Sans MS", 20, "bold"))
label0.pack()
label1 = tk.Label(image=test)
label1.image = test
label1.pack()

L2 = tk.Label(text="Answers: ", font = ("Comic Sans MS", 20, "bold"))
L2.pack()
for i in range(len(question)):
    Output = tk.Text(root, height = 3,
              width = 40,
              bg = "light cyan")
    Output.insert(tk.END, "Q: " + question[i]+"\n")
    Output.insert(tk.END, "A: " + answer[i])
    Output.pack()
    
root.mainloop()