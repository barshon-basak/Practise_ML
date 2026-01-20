#gradio app 

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("mobile_price_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_price(battery_power, blue, clock_speed, dual_sim, fc, four_g,
                  int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                  px_width, ram, sc_h, sc_w, talk_time, three_g, 
                  touch_screen, wifi):
    
    # Pack inputs into a DataFrame
    # The column names must match your dataset exactly
    data = {
        'battery_power': battery_power,
        'blue': blue,
        'clock_speed': clock_speed,
        'dual_sim': dual_sim,
        'fc': fc,
        'four_g': four_g,
        'int_memory': int_memory,
        'm_dep': m_dep,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'pc': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': sc_h,
        'sc_w': sc_w,
        'talk_time': talk_time,
        'three_g': three_g,
        'touch_screen': touch_screen,
        'wifi': wifi,
        # Engineered features
        'total_camera': pc + fc,
        'screen_area': sc_h * sc_w,
        'pixel_area': px_height * px_width,
        'battery_per_weight': battery_power / (mobile_wt + 1),
        'premium_features': blue + dual_sim + four_g + wifi
    }
    
    input_df = pd.DataFrame([data])
    
    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    # Return formatted result
    price_ranges = {
        0: 'Low Cost (₹5,000 - ₹10,000)',
        1: 'Medium Cost (₹10,000 - ₹20,000)',
        2: 'High Cost (₹20,000 - ₹30,000)',
        3: 'Very High Cost (₹30,000+)'
    }
    
    result = f"""
Predicted Price Range: {price_ranges[prediction]}

Confidence Scores:
  • Low Cost: {probabilities[0]*100:.1f}%
  • Medium Cost: {probabilities[1]*100:.1f}%
  • High Cost: {probabilities[2]*100:.1f}%
  • Very High Cost: {probabilities[3]*100:.1f}%
"""
    
    return result

# 3. The App Interface
# Defining inputs in a list to keep it clean
inputs = [
    gr.Slider(500, 2000, value=1000, label="Battery Power (mAh)"),
    gr.Radio([0, 1], value=1, label="Bluetooth"),
    gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Clock Speed (GHz)"),
    gr.Radio([0, 1], value=1, label="Dual SIM"),
    gr.Slider(0, 20, value=5, label="Front Camera (MP)"),
    gr.Radio([0, 1], value=1, label="4G"),
    gr.Slider(2, 128, value=32, label="Internal Memory (GB)"),
    gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Mobile Depth (cm)"),
    gr.Slider(80, 200, value=150, label="Mobile Weight (g)"),
    gr.Slider(1, 8, value=4, label="Number of Cores"),
    gr.Slider(0, 20, value=12, label="Primary Camera (MP)"),
    gr.Slider(0, 1960, value=720, label="Pixel Height"),
    gr.Slider(500, 1998, value=1080, label="Pixel Width"),
    gr.Slider(256, 4096, value=2048, label="RAM (MB)"),
    gr.Slider(5, 20, value=12, label="Screen Height (cm)"),
    gr.Slider(0, 18, value=6, label="Screen Width (cm)"),
    gr.Slider(2, 20, value=10, label="Talk Time (hours)"),
    gr.Radio([0, 1], value=1, label="3G"),
    gr.Radio([0, 1], value=1, label="Touch Screen"),
    gr.Radio([0, 1], value=1, label="WiFi")
]

app = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs="text", 
    title="📱 Mobile Price Range Predictor",
    description="Enter mobile specifications to predict the price range"
)

app.launch(share=True)

