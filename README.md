# 🌊 Water Scarcity Prediction System

A Machine Learning-based web application that predicts future water availability and classifies water scarcity levels (Low, Medium, High).

---

## 🚀 Overview

Water scarcity is increasing due to climate change, population growth, and rising water consumption. Traditional systems react late because they rely on manual monitoring and past data.

This project predicts water availability in advance using Machine Learning, helping in early decision-making and better water management.

---

## ⚙️ Features

- Predicts future water availability using Linear Regression  
- Classifies scarcity level (Low, Medium, High) using Random Forest  
- Uses key factors:
  - Rainfall (mm)
  - Temperature (°C)
  - Reservoir Levels (MCM)
  - Population
  - Water Consumption  
- Web-based interface for real-time prediction  

---

## 🧠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Flask  

---

## 📊 Model Performance

- Linear Regression: R² ≈ 0.85+  
- Random Forest: ~90% accuracy  
- Train-Test Split: 80-20  

---

## 📁 Project Structure
app.py
model/
templates/
data/

---

## ▶️ How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run the app:
python app.py

3. Open browser:
http://127.0.0.1:5000/

---

## 🎯 Goal

To predict water scarcity early and help in better planning and efficient water resource management.

---

