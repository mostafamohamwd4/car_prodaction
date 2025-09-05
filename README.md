#  Car Price Prediction App

A **Machine Learning powered web application** built with [Streamlit](https://streamlit.io/) that predicts the **resale price of cars** based on their features (mileage, engine capacity, fuel type, etc.).

---

##  Key Features
Interactive web interface using **Streamlit**  
Takes multiple car attributes as input (e.g., mileage, engine CC, seats, age)  
Converts user-friendly text input into numerical values automatically  
Shows the predicted price in a clean, formatted style  
Ready to integrate with any **trained ML model** (Joblib format)  

---

##  Input Features
The app collects the following features from the user:

| Feature         | Description                                      | Example            |
|-----------------|--------------------------------------------------|--------------------|
| Selling Price   | Current selling price (optional input)           | `450000`           |
| Km Driven       | Distance driven by the car                       | `145500`           |
| Fuel Type       | 1: Petrol, 2: Diesel, 3: CNG                     | `1`                |
| Seller Type     | 1: Individual, 2: Dealer, 3: Trustmark Dealer    | `2`                |
| Transmission    | 1: Manual, 2: Automatic                          | `1`                |
| Owner           | Number of previous owners                        | `0`                |
| Mileage         | Mileage (converted to float automatically)       | `20 kmpl` → `20.0` |
| Engine          | Engine displacement (CC, converted to float)     | `1200 CC` → `1200` |
| Max Power       | Maximum power (bhp, converted to float)          | `75 bhp` → `75.0`  |
| Seats           | Number of seats                                  | `5`                |
| Age             | Car age in years                                 | `5`                |
