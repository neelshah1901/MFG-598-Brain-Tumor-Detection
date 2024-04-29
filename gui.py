import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tkinter.font as tkFont

# Load your pre-trained model
model = load_model('/Users/neel/Desktop/model_after_training.h5')

def load_image():
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            loaded_image = Image.open(file_path)
            loaded_image = loaded_image.resize((224, 224), Image.Resampling.LANCZOS)
            img_display = ImageTk.PhotoImage(loaded_image)
            panel.configure(image=img_display)
            panel.image = img_display
            panel.pack(pady=20)

            # Preprocess the image to fit your model
            img_array = image.img_to_array(loaded_image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Ensure this matches your model's training preprocessing

            # Model prediction
            prediction = model.predict(img_array)
            if prediction.shape[1] == 1:  # Assuming binary classification with one output neuron
                result = int(prediction[0][0] > 0.5)
            else:
                result = np.argmax(prediction, axis=1)[0]

            if result == 0:
                result_label.config(text="No Tumor Detected", fg="green")
            else:
                result_label.config(text="Tumor Detected Action Required", fg="red")
            result_label.pack(pady=20)
        else:
            messagebox.showinfo("No file selected", "Please select an image file.")
    except Exception as e:
        messagebox.showerror("Error Loading Image", f"An error occurred: {e}")

def go_back():
    panel.pack_forget()
    result_label.pack_forget()
    title_label.pack()
    description_label.pack()
    btn.pack()

# Set up GUI
root = tk.Tk()
root.title("Brain Tumor Detection")

# Set up fonts and styles
title_font = tkFont.Font(family="Lucida Grande", size=25, weight="bold")
button_font = tkFont.Font(family="Lucida Grande", size=20)
result_font = tkFont.Font(family="Lucida Grande", size=20, weight="bold")

# Background color
root.configure(bg='lightblue')

# Set up the title and description labels
title_label = tk.Label(root, text="Brain Tumor Detector", font=title_font, bg='lightblue', fg='black')
title_label.pack(pady=20)
description_label = tk.Label(root, text="Upload an MRI image to detect whether a brain tumor is present or not.", bg='lightgray', fg='black')
description_label.pack(pady=10)

# Set up the image display panel
panel = tk.Label(root)
# Button to load image
btn = tk.Button(root, text='Load Image', command=load_image, font=button_font)
btn.pack(pady=20)

# Label to display results
result_label = tk.Label(root, text="Result", font=result_font)

# Add back button
back_btn = tk.Button(root, text='Back', command=go_back, font=button_font)

# Add menu
menu = tk.Menu(root)
root.config(menu=menu)
file_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open...", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Set window size and position
window_width = 1000
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

# Run the application
root.mainloop()


