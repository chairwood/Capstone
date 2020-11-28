from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk, Image

import yfinance as yf       #yfiance API: https://pypi.org/project/yfinance/
# import fix_yahoo_finance as fyf       #Unused as fix is the older lib

import tensorflow as tf
from keras.models import load_model
# from sklearn.externals
import joblib
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import date

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import pickle as pkl
import joblib
from sklearn.model_selection import train_test_split

plt.style.use('fivethirtyeight')



modelIndex = 0
scalerIndex = 0
scalers = ["TSLA_scaler", "GOOG_scaler", "MCD_scaler", "SBUX_scaler", "PFE_scaler", "NIO_scaler"]
options = ["TSLA_model.h5", "GOOG_model.h5", "MCD_model.h5", "SBUX_model.h5", "PFE_model.h5", "NIO_model.h5"]





root = Tk()
root.title("Stonks")
root.geometry("500x500")
# root.iconbitmap('deepLearning.ico')       #GUI Icon, fix later


#Popup Message Function
# showinfo, showwarning, showerror, askquestion, askokcancel, askyesno
def popup():
    response = messagebox.showinfo("This is my popup!", "Hello World!")
    # Label(root, text = response).grid(row=4, column=1)

#Popup Message Function
# def dataPopup():

    # Label(root, text=data).grid(row=6, column=0)
    # print(data['Close'])


    ## Alternative method, uses older lib: fix_yahoo_finance
    # nio = fyf.download('NIO', start='2017-01-01')
    # response = messagebox.showinfo("This is my popup!", nio['Close'])
    # Label(root, text=response).grid(row=0, column=3)

    # Label(root, text=nio['Close']).grid(row=0, column=3)
    # print(nio.head())

# # Clear Function
# def button_clear():
# 	myLabel.delete(0,END)


#Opens file for model; .h5
def openModel():
    global model_img
    # root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=(("png files", "*.png"), ("all files", "*.*")))

    scaler = joblib.load('NIO_scaler')
    print(scaler)



    # #Displays the model       #Helpful video how to save and load models: https://www.youtube.com/watch?v=7n1SpeudvAE
    new_model = load_model('NIO_model.h5')
    # my_img_label = Label(root, new_model.summary()).grid(row=2,column=3)      #Can't print to GUI
    print(new_model.summary())
    # print(new_model.get_weights())
    # print(new_model.optimizer)


    # #Prints directoy path
    # show_model = Label(root, text=root.filename).grid(row=1,column=3)
    # model_img = ImageTk.PhotoImage(Image.open(root.filename))
    #
    # #Displays the model
    # my_img_label = Label(image=model_img).grid(row=2,column=3)

#Show drop down selection
def show():
    # Selection = Label(root, text="Nothing Selected")
    # Selection.grid(row=3,column=3)
    # Selection.delete(0, END)
    Selection = Label(lookupFrame, text=clicked.get()).grid(row=6,column=1)

#Retrieves the data for a stock
def dataLookup():
    # stockEntryBox.get()
    # Selection = Label(lookupFrame, text=stockEntryBox.get()).grid(row=3, column=3)
    dataLabel = Label(lookupFrame, text=stockEntryBox.get()).grid(row=6,column=0)
    # dataPopup()

def graph():
    house_prices = np.random.normal(200000, 25000, 5000)
    plt.plot(house_prices, 50)
    plt.show()

def selectTSLA():
    global modelIndex
    modelIndex = 0
    scalerIndex = 0
    Selection = Label(graphFrame, text="TSLA").grid(row=2, column=0, columnspan=10)

def selectGOOG():
    global modelIndex
    global scalerIndex
    modelIndex = 1
    scalerIndex = 1
    Selection = Label(graphFrame, text="GOOG").grid(row=2, column=0, columnspan=10)

def selectMCD():
    global modelIndex
    global scalerIndex
    modelIndex = 2
    scalerIndex = 2
    Selection = Label(graphFrame, text="MCD").grid(row=2, column=0, columnspan=10)

def selectSBUX():
    global modelIndex
    global scalerIndex
    modelIndex = 3
    scalerIndex = 3
    Selection = Label(graphFrame, text="SBUX").grid(row=2, column=0, columnspan=10)

def selectPFE():
    global modelIndex
    global scalerIndex
    modelIndex = 4
    scalerIndex = 4
    Selection = Label(graphFrame, text="PFE").grid(row=2, column=0, columnspan=10)

def selectNIO():
    global modelIndex
    global scalerIndex
    modelIndex = 5
    scalerIndex = 5
    Selection = Label(graphFrame, text="NIO").grid(row=2, column=0, columnspan=10)



def predict():
    nio = yf.Ticker(stockEntryBox.get())
    # get stock data
    df = nio.history(period="max")
    data = df.filter(['Close'])
    dataset=data.values
    print(dataset)

    # Gets the number of rows to train with
    # Note: maybe changing the percentage will affect the accuracy ??
    training_data_length = math.ceil(len(dataset) * .8)
    # Prints the number of rows to give to the agent

    file_name = scalers[scalerIndex]
    file = open(file_name, 'rb')
    scaler = pkl.load(file)
    file.close()

    load_scaler = scaler
    scaled_data = load_scaler.fit_transform(data)

    # Collect all the values scaled data
    train_data = scaled_data[0:training_data_length, :]
    # Split the data in x_train and y_train data sets
    x_train = []
    y_train = []

    # try to decrease this window
    for i in range(60, len(train_data)):
        # Appends the 60 values before i, excluding i, for x_train
        x_train.append(train_data[i - 60:i])
        # Appends the ith value
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train2, y_train2 = np.array(x_train), np.array(y_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train2, y_train2, test_size=0.20, random_state=42)

#     load model

    new_model = load_model(options[modelIndex])


    # my_img_label = Label(root, new_model.summary()).grid(row=2,column=3)      #Can't print to GUI
    print(new_model.summary())



    test_data = scaled_data[training_data_length - 60:, :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_length:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])



    # Convert the sets into arrays
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = new_model.predict(x_test)
    # Kind of unscaling the dataset from earlier? Maybe try to leave the data unscaled?
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    print(rmse)

    # plot the data
    train = data[:training_data_length]
    valid = data[training_data_length:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16, 8))
    plt.title('Close Price History')
    plt.plot(df['Close'])  # Actual closing prices
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Close Price USD($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

    # Forcasting the next day's price using just the last 60 days

    today = date.today()

    # dd/mm/YY
    d1 = today.strftime("%Y-%m-%d")

    tesla_quote = nio.history(start='2020-9-27', end=d1)
    # create new dataframe
    new_df = tesla_quote.filter(['Close'])
    # get the last 60 dayt closing price values and covnert the dataframe to an array
    last_60_days = new_df[-60:].values
    # Scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    # Convert the set to an array
    X_test = np.array(X_test)
    # Reshape the data
    # Needs to be 3 elements
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # Get the predicted scaled price
    pred_price = new_model.predict(X_test)
    # undo the scaling
    pred_price = scaler.inverse_transform(pred_price)

    predictionLabel = Label(root, text=pred_price)
    predictionLabel.grid(row=4, column=1)
    print(pred_price)




#NEURAL NETWORKING LOGO
my_img = ImageTk.PhotoImage(Image.open("deepLearning.png"))
appPNG = Label(image=my_img)
appPNG.grid(row=0, column=0, columnspan=20)

#CAPSTONE TITLE
appTitle = Label(root, text = "Diep Learning Stock Forecaster")
appTitle.grid(row=1, column=0, columnspan=20)



#LEFT FRAME
lookupFrame = LabelFrame(root, text = "Input", padx=5, pady=25)
lookupFrame.grid(row=3, column=0)


stockEntryBoxTitle = Label(lookupFrame, text = "Stock Symbol")
stockEntryBoxTitle.grid(row=0, column=0, sticky=W, columnspan=2)
stockEntryBox = Entry(lookupFrame, width = 20, borderwidth = 5)
stockEntryBox.grid(row=1, column=0, columnspan=2)
# stockLookupButton = Button(lookupFrame, text="Lookup Data", command = dataLookup)
# stockLookupButton.grid(row=2, column=0, columnspan=2)

#DROP DOWN BOXES
# clicked = StringVar()
# clicked.set(options[0])
# modelSelection = Label(lookupFrame, text = "Model Selection")
# modelSelection.grid(row=3, column=0, sticky=W, columnspan=2)
# drop = OptionMenu(lookupFrame, clicked, *options)
# drop.grid(row=4, column=0, columnspan=2)

predictButton = Button(lookupFrame, text="Predict", command = predict)
predictButton.grid(row=5, column=0, columnspan=2)

# dropDownButton = Button(lookupFrame, text="Confirm", command=show).grid(row=5,column=0, columnspan=2)




#RIGHT FRAME
graphFrame = LabelFrame(root, text = "Model Selection", padx=25, pady=25)
graphFrame.grid(row=3, column=1)

#OPENING FILE 1
# modelButton = Button(graphFrame, text="Open File", command = openModel)
# modelButton.grid(row=4,column=4)

# df = web.DataReader('NIO', data_source='yahoo', start='2018-11-01', end='2020-11-03')


b1 = Button(graphFrame, text="TSLA", command = selectTSLA).grid(row=0,column=0)
b2 = Button(graphFrame, text="GOOG", command = selectGOOG).grid(row=0,column=1)
b3 = Button(graphFrame, text="MCD", command = selectMCD).grid(row=0,column=2)
b4 = Button(graphFrame, text="SBUX", command = selectSBUX).grid(row=0,column=3)
b5 = Button(graphFrame, text="PFE", command = selectPFE).grid(row=0,column=4)
b6 = Button(graphFrame, text="NIO", command = selectNIO).grid(row=0,column=5)

modelSelection = Label(graphFrame, text = "Model Selection").grid(row=1, column=0, columnspan= 20)
# b9 = Button(graphFrame, text="Popup Message", command=popup)
# b4 = Button(graphFrame, text="Show me the data!", command=dataPopup)
# b9.grid(row=2,column=2)
# b4.grid(row=3,column=3)


#PREDICTION
predictionTitle = Label(root, text = "Prediction: ")
predictionTitle.grid(row=4, column=0, sticky=W)




#QUIT BUTTON
button_quit = Button(root, text="Exit Program", command=root.quit, bg ="#FF0000")
button_quit.grid(row=10, column=0, columnspan=20, sticky=W+E)

# status = Label(root, text="Image 1 of 5", bd=1, relief=SUNKEN, anchor=E)
# status.grid(row=2, column=0, columnspan=2)





root.mainloop()