import tkinter as tk
import LSTMPrediction
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import pandas_datareader.data as web
import subprocess
from tkinter import *

root = tk.Tk()
root.title("Name of App")
model_recommendation = tk.StringVar()
label_stock_name = tk.StringVar()
label_stock_name.set('STOCK')
company_name = tk.StringVar()
company_name.set("Stock")

def get_company_name(ticker_symbol):
    try:
        company = web.DataReader(ticker_symbol, 'yahoo')
        return company.loc[company.index[0], 'Name']
    except:
        return None

#Get input text from the search bar
def get_stock_name():
    stock_name = search_bar.get()
    label_stock_name.set(stock_name)
    company_name = get_company_name(label_stock_name)
    stock_plot, model_recommendation = LSTMPrediction.predict_stock(stock_name)
    print(model_recommendation)
    plot_stock_trend(stock_plot)


def open_chat():
    subprocess.run(["python", "chat.py"])    

#Draw the graph on the canvas
def plot_stock_trend(stock_plot):
    canvas = FigureCanvasTkAgg(stock_plot, master=root)
    canvas.draw()
      
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()

    canvas.get_tk_widget().place(relx=0.35, rely=0.3, anchor='nw')
    canvas.get_tk_widget().config(width=500, height=400)
    #canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)  

def set_favourite():
    stock_name = search_bar.get()
    if stock_name in favourite_list.get(0,"end"):
        remove_index = favourite_list.get(0,tk.END).index(stock_name)
        favourite_list.delete(remove_index)
        print(favourite_list)
    else:
        favourite_list.insert(tk.END, stock_name)
        print(favourite_list)
    
def get_top_stocks():
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"]
    sp500 = sp500.head(10)
    return sp500    

def get_active_stock():
    top_stock = top_stock_list.get(ACTIVE)
    stock_plot, model_recommendation = LSTMPrediction.predict_stock(top_stock)
    print(model_recommendation)
    plot_stock_trend(stock_plot)   

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size to half of the screen size
root.geometry(f"{int(screen_width/2)}x{int(screen_height/2)}")

# Change the background color to #BB6161 
root.config(bg='#BB6161')

# Create a new frame widget for the middle frame
middle_frame = tk.Frame(root, bg='#EEE8E8')
middle_frame.place(relx=0.5, rely=0.5, relwidth=0.9, relheight=0.9, anchor='center')

# Create a new frame widget for the left frame
left_frame = tk.Frame(middle_frame, bg='#D2CCCC')
left_frame.place(relx=0, rely=0.5, relwidth=0.3, relheight=1, anchor='w')

# Add label for "Name of App"
app_name = tk.Label(left_frame, text="Name of App", font=("Helvetica", 16, "bold"), bg="#D2CCCC", fg="#000000")
app_name.place(relx=0, rely=0, relwidth=1, relheight=0.15)

# Add label for "Top Stock"
top_stock = tk.Label(left_frame, text="Top Stock", font=("Helvetica", 14), bg="#D2CCCC", fg="#706C6B")
top_stock.place(relx=0, rely=0.25, relwidth=1, relheight=0.05)

# Add list of example stocks
#stock_list = tk.Label(left_frame, text="Example Stock 1\nExample Stock 2", font=("Helvetica", 12), bg="#D2CCCC", fg="#F6F2F1")
#stock_list.place(relx=0, rely=0.30, relwidth=1, relheight=0.2)

# Add label for "Favourites"
favourites = tk.Label(left_frame, text="Favourites", font=("Helvetica", 14), bg="#D2CCCC", fg="#706C6B")
favourites.place(relx=0, rely=0.60, relwidth=1, relheight=0.05)

# Add list of example favourites
#favourite_list = tk.Label(left_frame, text="Example Favourite 1\nExample Favourite 2", font=("Helvetica", 12), bg="#D2CCCC", fg="#F6F2F1")
#favourite_list.place(relx=0, rely=0.65, relwidth=1, relheight=0.2)

# Add horizontal line
line = tk.Frame(left_frame, bg="#706C6B", height=0.005)
line.place(relx=0, rely=0.9, relwidth=1, relheight=0.03)

right_frame = tk.Frame(middle_frame, bg='#EEE8E8')
right_frame.place(relx=0.3, rely=0.5, relwidth=1, relheight=1, anchor='w')



# create button on right frame
home_button = tk.Button(right_frame, text="Home", font=("Helvetica", 16, "bold"), bg="#D2CCCC", fg="#000000")
home_button.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')


search_bar = tk.Entry(right_frame, bg='#FFF5F4',width=50)
search_bar.insert(0, "search")
search_bar.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')

search_button = tk.Button(right_frame, text = "Search", font = ("Helvetica", 16, "bold"), bg = '#D2CCCC', fg = '#000000', command = get_stock_name)
search_button.grid(row = 0, column = 5, padx = 0, pady = 5, sticky = 'nsew')



# create text "STOCK" on left side of right frame
stock_text = tk.Label(right_frame, textvariable=label_stock_name.get(), font=("bold"), fg='black', bg='#FFF5F4')
stock_text.place(relx=0.1, rely=0.2, anchor='nw')

# create placeholder for graph on right frame
graph_placeholder = tk.Label(right_frame, text="Graph Pending", fg='black', bg='#FFF5F4')
graph_placeholder.place(relx=0.3, rely=0.5, anchor='nw')

# create text "Model Recommendation" on right frame
model_text = tk.Label(right_frame, text=('Model Recommendation: ', model_recommendation), fg='black', bg='#FFF5F4')
model_text.place(relx=0.25, rely=0.9, anchor='nw')

#creating scrollbar adn list box for favourites
scrollbar = tk.Scrollbar(left_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

favourites_int = tk.IntVar()
favourite_list = tk.Listbox(left_frame, yscrollcommand= scrollbar.set)
favourite_list.__contains__ = lambda str:str in favourite_list.get(0,"end")
favourite_list.place(relx=0, rely=0.65, relwidth=1, relheight=0.2)

# create favourites button
favourites_button = tk.Button(right_frame, text="+", font=("Helvetica", 16, "bold"), bg="#D2CCCC", fg="#000000", command = set_favourite)
favourites_button.place(relx = 0.6, rely = 0.2, anchor = 'nw')


root.mainloop()
