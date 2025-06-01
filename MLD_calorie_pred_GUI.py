import torch
import torch.nn as nn
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import sv_ttk
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Define your model class (same architecture as the one you trained)
# R2 score of approx .99 on val data , closer to 1 better it is !

class CaloriePredictor(nn.Module):
    def __init__(self):
        super(CaloriePredictor, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(7,56),
            nn.ReLU(),
            nn.Linear(56,224),
            nn.ReLU(),
            nn.Linear(224,896),
            nn.ReLU(),
            nn.Linear(896,448),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(448,112),
            nn.ReLU(),
            nn.Linear(112,64),
            nn.ReLU(),
            nn.Dropout(.2),            
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,8),
            nn.ReLU(),
            nn.Linear(8,1),
        )

    def forward(self, x):
        return self.net(x)

# Load model and scalers with weights_only=True for security
model = CaloriePredictor()
model.load_state_dict(torch.load("Calories_Prediction.pt", weights_only=True))
model.eval()
X_scaler = joblib.load("X_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

class CaloriePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Calorie Burn Predictor Pro")
        self.root.geometry("900x700")
        self.root.minsize(800, 650)
        
        # Set theme (using sv_ttk for modern look)
        sv_ttk.set_theme("dark")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#333333')
        self.style.configure('TLabel', background='#333333', foreground='white', font=('Helvetica', 10))
        self.style.configure('TButton', font=('Helvetica', 10, 'bold'))
        self.style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'))
        self.style.configure('Result.TLabel', font=('Helvetica', 24, 'bold'), foreground='#4CAF50')
        self.style.configure('Footer.TLabel', font=('Helvetica', 8), foreground='#AAAAAA')
        
        # Create main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header section
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.header_label = ttk.Label(
            self.header_frame, 
            text="CALORIE BURN PREDICTOR", 
            style='Header.TLabel'
        )
        self.header_label.pack(side=tk.LEFT)
        
        # Add logo (placeholder - replace with your own image)
        try:
            self.logo_img = Image.open("logo.png").resize((40, 40))
            self.logo_photo = ImageTk.PhotoImage(self.logo_img)
            self.logo_label = ttk.Label(self.header_frame, image=self.logo_photo)
            self.logo_label.pack(side=tk.RIGHT, padx=10)
        except:
            pass
        
        # Input section
        self.input_frame = ttk.LabelFrame(
            self.main_frame, 
            text="Activity Details", 
            padding=(20, 10)
        )
        self.input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create input fields
        self.create_input_fields()
        
        # Result section
        self.result_frame = ttk.Frame(self.main_frame)
        self.result_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.result_label = ttk.Label(
            self.result_frame, 
            text="Predicted Calories Burned:", 
            style='Header.TLabel'
        )
        self.result_label.pack(side=tk.LEFT)
        
        self.result_value = ttk.Label(
            self.result_frame, 
            text="0.00 cal", 
            style='Result.TLabel'
        )
        self.result_value.pack(side=tk.RIGHT)
        
        # Button section
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.predict_btn = ttk.Button(
            self.button_frame,
            text="CALCULATE CALORIES",
            command=self.predict,
            style='Accent.TButton'
        )
        self.predict_btn.pack(fill=tk.X, ipady=10)
        
        # Footer section with your name and email
        self.footer_frame = ttk.Frame(self.main_frame)
        self.footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Add your name and email in the footer (right-aligned)
        self.footer_label = ttk.Label(
            self.footer_frame,
            text="Created by Souradeep Dutta | Email : aimldatasets22@gmail.com",
            style='Footer.TLabel'
        )
        self.footer_label.pack(side=tk.RIGHT)
        
        # Status bar
        self.status_bar = ttk.Label(
            self.main_frame, 
            text="Ready", 
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, pady=(10, 0))
        
    def create_input_fields(self):
        # Personal info
        personal_frame = ttk.Frame(self.input_frame)
        personal_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(personal_frame, text="Personal Information").pack(anchor=tk.W)
        
        # Sex
        sex_frame = ttk.Frame(personal_frame)
        sex_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sex_frame, text="Sex:").pack(side=tk.LEFT, padx=(0, 10))
        self.sex_var = tk.StringVar(value="Male")
        self.male_radio = ttk.Radiobutton(
            sex_frame, 
            text="Male", 
            variable=self.sex_var, 
            value="Male"
        )
        self.male_radio.pack(side=tk.LEFT)
        self.female_radio = ttk.Radiobutton(
            sex_frame, 
            text="Female", 
            variable=self.sex_var, 
            value="Female"
        )
        self.female_radio.pack(side=tk.LEFT, padx=(10, 0))
        
        # Age
        age_frame = ttk.Frame(personal_frame)
        age_frame.pack(fill=tk.X, pady=5)
        ttk.Label(age_frame, text="Age (years):").pack(side=tk.LEFT)
        self.age_entry = ttk.Entry(age_frame)
        self.age_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Height
        height_frame = ttk.Frame(personal_frame)
        height_frame.pack(fill=tk.X, pady=5)
        ttk.Label(height_frame, text="Height (cm):").pack(side=tk.LEFT)
        self.height_entry = ttk.Entry(height_frame)
        self.height_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Weight
        weight_frame = ttk.Frame(personal_frame)
        weight_frame.pack(fill=tk.X, pady=5)
        ttk.Label(weight_frame, text="Weight (kg):").pack(side=tk.LEFT)
        self.weight_entry = ttk.Entry(weight_frame)
        self.weight_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Activity info
        activity_frame = ttk.Frame(self.input_frame)
        activity_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Label(activity_frame, text="Activity Information").pack(anchor=tk.W)
        
        # Duration
        duration_frame = ttk.Frame(activity_frame)
        duration_frame.pack(fill=tk.X, pady=5)
        ttk.Label(duration_frame, text="Duration (minutes):").pack(side=tk.LEFT)
        self.duration_entry = ttk.Entry(duration_frame)
        self.duration_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Heart Rate
        hr_frame = ttk.Frame(activity_frame)
        hr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(hr_frame, text="Heart Rate (bpm):").pack(side=tk.LEFT)
        self.heart_rate_entry = ttk.Entry(hr_frame)
        self.heart_rate_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Body Temp
        temp_frame = ttk.Frame(activity_frame)
        temp_frame.pack(fill=tk.X, pady=5)
        ttk.Label(temp_frame, text="Body Temperature (°C):").pack(side=tk.LEFT)
        self.body_temp_entry = ttk.Entry(temp_frame)
        self.body_temp_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Set default values for testing
        self.age_entry.insert(0, "25")
        self.height_entry.insert(0, "175")
        self.weight_entry.insert(0, "70")
        self.duration_entry.insert(0, "30")
        self.heart_rate_entry.insert(0, "120")
        self.body_temp_entry.insert(0, "37")
    
    def predict(self):
        try:
            self.status_bar.config(text="Processing...")
            self.root.update()
            
            sex = 1 if self.sex_var.get().lower() == 'male' else 0
            age = float(self.age_entry.get())
            height = float(self.height_entry.get())
            weight = float(self.weight_entry.get())
            duration = float(self.duration_entry.get())
            heart_rate = float(self.heart_rate_entry.get())
            body_temp = float(self.body_temp_entry.get())
            
            # Validate inputs
            if not (0 < age < 120):
                raise ValueError("Please enter a valid age (0-120)")
            if not (100 < height < 250):
                raise ValueError("Please enter a valid height (100-250 cm)")
            if not (30 < weight < 300):
                raise ValueError("Please enter a valid weight (30-300 kg)")
            if not (1 <= duration <= 600):
                raise ValueError("Please enter a valid duration (1-600 minutes)")
            if not (40 <= heart_rate <= 220):
                raise ValueError("Please enter a valid heart rate (40-220 bpm)")
            if not (35 <= body_temp <= 42):
                raise ValueError("Please enter a valid body temperature (35-42°C)")
            
            features = [sex, age, height, weight, duration, heart_rate, body_temp]
            
            # Normalize input
            input_np = np.array(features).reshape(1, -1)
            input_scaled = X_scaler.transform(input_np)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float)
            
            # Predict
            with torch.no_grad():
                pred_norm = model(input_tensor).numpy()
            
            # Denormalize
            pred_actual = y_scaler.inverse_transform(pred_norm)
            result = float(pred_actual[0][0])
            
            # Update result display with animation
            self.animate_result(result)
            self.status_bar.config(text="Prediction complete")
            
        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
            self.status_bar.config(text="Error in input values")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_bar.config(text="Error occurred")
    
    def animate_result(self, final_value):
        """Animate the result display counting up"""
        current = 0
        step = max(1, final_value / 20)  # Calculate step size for smooth animation
        
        def update():
            nonlocal current
            if current < final_value:
                current += step
                if current > final_value:
                    current = final_value
                self.result_value.config(text=f"{current:.2f} cal")
                self.root.after(20, update)
            else:
                self.result_value.config(text=f"{final_value:.2f} cal")
        
        update()

if __name__ == "__main__":
    root = tk.Tk()
    app = CaloriePredictorApp(root)
    root.mainloop()