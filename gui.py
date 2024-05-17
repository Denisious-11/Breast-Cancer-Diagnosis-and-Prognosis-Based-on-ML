from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import re
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd
import joblib

#Diagnosis
d_classifier_lr = joblib.load('models/Diagnostic/classifier_lr_model.joblib')
d_classifier_svc = joblib.load('models/Diagnostic/classifier_svc_model.joblib')
d_classifier_tree = joblib.load('models/Diagnostic/classifier_tree_model.joblib')
d_classifier_nb = joblib.load('models/Diagnostic/classifier_nb_model.joblib')
d_ensemble_classifier = joblib.load('models/Diagnostic/ensemble_classifier_model.joblib')
d_mms = joblib.load('models/Diagnostic/minmaxscaler_model.joblib')
d_ann_model = joblib.load('models/Diagnostic/ann_model.joblib')

#Prognosis
p_classifier_lr = joblib.load('models/Prognostic/classifier_lr_model.joblib')
p_classifier_svc = joblib.load('models/Prognostic/classifier_svc_model.joblib')
p_classifier_tree = joblib.load('models/Prognostic/classifier_tree_model.joblib')
p_classifier_nb = joblib.load('models/Prognostic/classifier_nb_model.joblib')
p_ensemble_classifier = joblib.load('models/Prognostic/ensemble_classifier_model.joblib')
p_mms = joblib.load('models/Prognostic/minmaxscaler_model_.joblib')
p_ann_model = joblib.load('models/Prognostic/ann_model_.joblib')


#window creation
a = Tk()
a.title("FocusMammo")
a.geometry("1000x500")
a.minsize(1000,500)
a.maxsize(1000,500)


def prediction():
    
    alltext=text1.get("1.0",'end')
    if alltext=='' or alltext=='\n':
        message.set("fill the empty field!!!")
    else:
        list_box.insert(1, "Preprocessing")
        list_box.insert(2, "")
        list_box.insert(3, "Perform Feature Scaling")
        list_box.insert(4, "")
        list_box.insert(5, "Loading Diagnosis Model")
        list_box.insert(6, "")
        list_box.insert(7, "Prediction")
        message.set("")

        relist=[]
        list1=alltext.split(",")
        print(list1)
        floatlist=[float(x)for x in list1]
        print(floatlist)

        new_data_array = np.array(floatlist).reshape(1, -1)

        # Create a DataFrame using the reshaped array
        new_data = pd.DataFrame(new_data_array, columns=[
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
            'concavity_mean', 'concave points_mean', 'symmetry_mean', 'radius_se', 
            'perimeter_se', 'area_se', 'compactness_se', 'concavity_se', 'concave points_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ])


        # Scale the new data using the MinMaxScaler
        cp = d_mms.transform(new_data)

        lr_predictions = d_classifier_lr.predict(cp)
        svc_predictions = d_classifier_svc.predict(cp)
        tree_predictions = d_classifier_tree.predict(cp)
        nb_predictions = d_classifier_nb.predict(cp)

        # Concatenate predictions
        cp_ = np.column_stack((lr_predictions, svc_predictions, tree_predictions, nb_predictions))
        print("Concatenated Predictions:", cp_)

        # Make predictions using the ANN model
        ann_predictions = d_ann_model.predict(cp_)

        # Print or use the predictions as needed

        print("ANN Predictions:", ann_predictions)

        if ann_predictions==0:
            print("\n[Result] : Benign")
            output="Benign"
        if ann_predictions==1:
            print("\n[Result] : Cancer Detected")
            output="Cancer Detected"

        out_label.config(text=output)


def prediction2():
    
    alltext=text1.get("1.0",'end')
    if alltext=='' or alltext=='\n':
        message.set("fill the empty field!!!")
    else:
        list_box.insert(1, "Preprocessing")
        list_box.insert(2, "")
        list_box.insert(3, "Perform Feature Scaling")
        list_box.insert(4, "")
        list_box.insert(5, "Loading Prognosis Model")
        list_box.insert(6, "")
        list_box.insert(7, "Prediction")
        message.set("")

        relist=[]
        list1=alltext.split(",")
        print(list1)
        floatlist=[float(x)for x in list1]
        print(floatlist)

        new_data_array = np.array(floatlist).reshape(1, -1)

        # Create a DataFrame using the reshaped array
        new_data = pd.DataFrame(new_data_array, columns=[
            'Radius_mean','Perimeter_mean','Area_mean','Smoothness_mean','Compactness_mean','Concavity_mean',
            'Concave_points_mean','Radius_SE','Perimeter_SE','Area_SE','Compactness_SE','Symmetry_SE',
            'Fractal_dimension_SE','Radius_worst','Texture_worst','Perimeter_worst','Area_worst',
            'Smoothness_worst','Compactness_worst','Concavity_worst','Concave_points_worst','Fractal_dimension_worst',
            'Tumor_size','Lymph_node_status'
        ])


        # Scale the new data using the MinMaxScaler
        cp = p_mms.transform(new_data)

        # Make predictions using classifiers
        lr_predictions = p_classifier_lr.predict(cp)
        svc_predictions = p_classifier_svc.predict(cp)
        tree_predictions = p_classifier_tree.predict(cp)
        nb_predictions = p_classifier_nb.predict(cp)

        # Concatenate predictions
        cp_ = np.column_stack((lr_predictions, svc_predictions, tree_predictions, nb_predictions))
        print("Concatenated Predictions:", cp_)

        # Make predictions using the ANN model
        ann_predictions = p_ann_model.predict(cp)

        # Print or use the predictions as needed

        print("ANN Predictions:", ann_predictions)

        if ann_predictions==0:
            print("\n[Result] : Non-Recurrent")
            output="Non-Recurrent"
        if ann_predictions==1:
            print("\n[Result] : Recurrent")
            output="Recurrent"


        out_label.config(text=output)



def Check():
    global f
    f.pack_forget()

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="#A7BEAE")
    f1.place(x=0, y=0, width=760, height=250)
    f1.config()

    input_label = Label(f1, text="Diagnosis: INPUT", font="arial 16", bg="#A7BEAE")
    input_label.pack(padx=0, pady=10)

    
    global message
    message = StringVar()

    global text1
    text1=Text(f1,height=8,width=70)
    text1.pack()


    msg_label = Label(f1, text=
        "", textvariable=message,
                      bg='#A7BEAE').place(x=330, y=185)

    predict_button = Button(
        f1, text="Predict", command=prediction, bg="yellow")
    predict_button.pack(side="bottom", pady=16)
    global f2
    f2 = Frame(f, bg="#E7E8D1")
    f2.place(x=0, y=250, width=760, height=500)
    f2.config(pady=20)

    result_label = Label(f2, text="RESULT", font="arial 16", bg="#E7E8D1")
    result_label.pack(padx=0, pady=0)

    global out_label
    out_label = Label(f2, text="", bg="#E7E8D1", font="arial 16")
    out_label.pack(pady=70)

    f3 = Frame(f, bg="#C5FAD5")
    f3.place(x=760, y=0, width=240, height=690)
    f3.config()

    name_label = Label(f3, text="PROCESS", font="arial 14", bg="#C5FAD5")
    name_label.pack(pady=20)

    global list_box
    list_box = Listbox(f3, height=12, width=31)
    list_box.pack()





def Check2():
    global f
    f.pack_forget()

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="#A7BEAE")
    f1.place(x=0, y=0, width=760, height=250)
    f1.config()

    input_label = Label(f1, text="Prognosis: INPUT", font="arial 16", bg="#A7BEAE")
    input_label.pack(padx=0, pady=10)

    
    global message
    message = StringVar()

    global text1
    text1=Text(f1,height=8,width=70)
    text1.pack()


    msg_label = Label(f1, text=
        "", textvariable=message,
                      bg='#A7BEAE').place(x=330, y=185)

    predict_button = Button(
        f1, text="Predict", command=prediction2, bg="yellow")
    predict_button.pack(side="bottom", pady=16)
    global f2
    f2 = Frame(f, bg="#E7E8D1")
    f2.place(x=0, y=250, width=760, height=500)
    f2.config(pady=20)

    result_label = Label(f2, text="RESULT", font="arial 16", bg="#E7E8D1")
    result_label.pack(padx=0, pady=0)

    global out_label
    out_label = Label(f2, text="", bg="#E7E8D1", font="arial 16")
    out_label.pack(pady=70)

    f3 = Frame(f, bg="#C5FAD5")
    f3.place(x=760, y=0, width=240, height=690)
    f3.config()

    name_label = Label(f3, text="PROCESS", font="arial 14", bg="#C5FAD5")
    name_label.pack(pady=20)

    global list_box
    list_box = Listbox(f3, height=12, width=31)
    list_box.pack()



def Home():
    global f
    f.pack_forget()

    f = Frame(a, bg="light goldenrod")
    f.pack(side="top", fill="both", expand=True)

    front_image = Image.open("Extra/home.jpg")
    front_photo = ImageTk.PhotoImage(front_image.resize((a.winfo_width(), a.winfo_height()), Image.ANTIALIAS))
    front_label = Label(f, image=front_photo)
    front_label.image = front_photo
    front_label.pack()

    home_label = Label(f, text="Breast Cancer Diagnosis & Prognosis",
                       font="arial 35", bg="light goldenrod")
    home_label.place(x=120, y=200)


f = Frame(a, bg="light goldenrod")
f.pack(side="top", fill="both", expand=True)

front_image1 = Image.open("Extra/home.jpg")
front_photo1 = ImageTk.PhotoImage(front_image1.resize((1000,650), Image.ANTIALIAS))
front_label1 = Label(f, image=front_photo1)
front_label1.image = front_photo1
front_label1.pack()

home_label = Label(f, text="Breast Cancer Diagnosis & Prognosis",
                   font="arial 35", bg="light goldenrod")
home_label.place(x=120, y=200)

m = Menu(a)
m.add_command(label="Homepage", command=Home)
checkmenu = Menu(m)
m.add_command(label="Diagnosis", command=Check)
m.add_command(label="Prognosis", command=Check2)
plotmenu=Menu(m)
a.config(menu=m)


a.mainloop()
