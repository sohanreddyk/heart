from tkinter import Tk, Label, Button, Entry, messagebox
from tkinter import ttk
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Ensure Matplotlib uses the TkAgg backend for compatibility with Tkinter
matplotlib.use("TkAgg")

# Load models (ensure the models are in the correct directory)
models = {
    "Random Forest": "heart_disease_random_forest_model.pkl",
    "Logistic Regression": "heart_disease_logistic_regression_model.pkl",
    "SVM": "heart_disease_svm_model.pkl",
    "KNN": "heart_disease_knn_model.pkl",
}

# Load the best model based on the backend selection
best_model_name = "Random Forest"  # Replace with the model you want to use based on performance
best_model_path = models[best_model_name]

if os.path.exists(best_model_path):
    best_model = joblib.load(best_model_path)
else:
    messagebox.showerror("Error", f"{best_model_path} does not exist!")
    exit()

# Function to make predictions
def predict_prognosis():
    try:
        # Fetch input data from the GUI
        age = int(age_entry.get())
        sex = int(sex_combobox.get())
        cp = int(cp_combobox.get())
        trestbps = int(trestbps_entry.get())
        chol = int(cholesterol_entry.get())
        fbs = int(fbs_combobox.get())
        restecg = int(restecg_combobox.get())
        thalach = int(thalach_entry.get())
        exang = int(exang_combobox.get())
        oldpeak = float(oldpeak_entry.get())
        slope = int(slope_combobox.get())
        ca = int(ca_combobox.get())
        thal = int(thal_combobox.get())

        # Convert input data to DataFrame for prediction
        input_data = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, 
            restecg, thalach, exang, oldpeak, slope, ca, thal
        ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                     'ca', 'thal'])

        # Predict the result using the best model
        prediction = best_model.predict(input_data)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        color = "red" if prediction == 1 else "green"

        # Update the result label
        result_label.config(text=f"Predicted Prognosis: {result}", fg=color, font=("Arial", 14, "bold"))

        # Here, we assume the true label (ground truth) for the input
        # For testing purposes, you can set this value manually or get it from your dataset
        # Here, 1 means heart disease, 0 means no disease (modify as per your actual ground truth)
        true_label = 1  # Set actual value as per your data (use 1 for "Heart Disease Detected", 0 for "No Heart Disease")

        # Plot confusion matrix with the correct ground truth and prediction
        plot_confusion_matrix(best_model, input_data, true_label)

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for all fields.")
        
# Function to plot the confusion matrix and display graph
def plot_confusion_matrix(model, input_data, true_label):
    # Get the prediction from the model
    y_pred = model.predict(input_data)

    # Generate confusion matrix
    cm = confusion_matrix([true_label], y_pred)  # Use true_label for ground truth and y_pred for prediction

    # Create confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"], ax=ax)
    ax.set_title(f"Confusion Matrix for {best_model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Embed the plot into the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().place(x=700, y=100)  # Position the graph on the right side
    canvas.draw()



# Initialize the Tkinter GUI
root = Tk()
root.title("Heart Disease Prediction System")
root.geometry("1000x850")
root.config(bg="#f0f0f0")

# Title label
title_label = Label(root, text="Heart Disease Prediction System", font=("Arial", 20, "bold"), bg="#62a7ff", fg="white")
title_label.pack(fill="both", pady=10)

# Input fields and labels (same as before)
input_labels = [
    ("Age", 100), ("Sex (1=Male, 0=Female)", 150), ("Chest Pain Type (cp)", 200),
    ("Resting Blood Pressure (trestbps)", 250), ("Cholesterol (chol)", 300),
    ("Fasting Blood Sugar (fbs)", 350), ("Resting Electrocardiogram (restecg)", 400),
    ("Max Heart Rate (thalach)", 450), ("Exercise Induced Angina (exang)", 500),
    ("Oldpeak", 550), ("Slope", 600), ("Number of Major Vessels (ca)", 650),
    ("Thalassemia (thal)", 700),
]

input_fields = []
for label, y in input_labels:
    Label(root, text=label, font=("Arial", 12), bg="#f0f0f0").place(x=50, y=y)
    if label in ["Age", "Resting Blood Pressure (trestbps)", "Cholesterol (chol)", "Max Heart Rate (thalach)", "Oldpeak"]:
        entry = Entry(root, font=("Arial", 12), width=25)
        entry.place(x=300, y=y)
        input_fields.append(entry)
    else:
        values = {
            "Sex (1=Male, 0=Female)": [0, 1],
            "Chest Pain Type (cp)": [0, 1, 2, 3],
            "Fasting Blood Sugar (fbs)": [0, 1],
            "Resting Electrocardiogram (restecg)": [0, 1, 2],
            "Exercise Induced Angina (exang)": [0, 1],
            "Slope": [0, 1, 2],
            "Number of Major Vessels (ca)": [0, 1, 2, 3],
            "Thalassemia (thal)": [3, 6, 7],
        }
        combobox = ttk.Combobox(root, values=values[label], font=("Arial", 12), width=22)
        combobox.place(x=300, y=y)
        input_fields.append(combobox)

# Map input fields to variables
(age_entry, sex_combobox, cp_combobox, trestbps_entry, cholesterol_entry, 
 fbs_combobox, restecg_combobox, thalach_entry, exang_combobox, oldpeak_entry, 
 slope_combobox, ca_combobox, thal_combobox) = input_fields

# Predict button
predict_button = Button(root, text="Predict Prognosis", font=("Arial", 14), bg="#62a7ff", fg="white", command=predict_prognosis)
predict_button.place(x=250, y=750, width=300, height=40)

# Result label
result_label = Label(root, text="", font=("Arial", 14), bg="#f0f0f0", fg="black")
result_label.place(x=700, y=550)  # Position the result below the graph on the right side

# Run the application
root.mainloop()
