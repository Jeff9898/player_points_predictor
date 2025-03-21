# Fantasy Premier League (FPL) Player Points Prediction Tool

This project provides a data-driven tool designed to predict Fantasy Premier League (FPL) player points for upcoming gameweeks. Built with Python and Streamlit, the tool leverages historical player performance data and machine learning to provide helpful and accurate predictions. The goal is to assist fantasy soccer (football, depending where you're from) managers in making informed decisions for their teams line-ups and transfers, ultimately optimizing their performance and rankings.

---

## Key Features

- **Predictive Modeling:** Utilizes a Random Forest regression model to forecast FPL player points.
- **Interactive Dashboard:** Built using Streamlit, enabling users to visualize predictions, compare players, and interact with performance data.
- **Exploratory Data Analysis:** Offers insights through meaningful visualizations such as scatter plots, regression plots, and histograms.

---

## Technologies Used

- **Python** – Core programming language
- **Pandas & NumPy** – Data manipulation and preprocessing
- **Scikit-learn** – Machine learning modeling and evaluation
- **Streamlit** – User interface and interactive dashboard
- **Matplotlib & Seaborn** – Visualization libraries for data exploration

---

## Data Source

The data for this project is publicly available and sourced from [Vaastav's Fantasy Premier League GitHub repository](https://github.com/vaastav/Fantasy-Premier-League)

---

## Installation & Local Usage

### Step 1: Clone the Repository

```shell
git clone https://github.com/Jeff9898/player_points_predictor.git

```

### Step 2: Set Up Python Environment

Ensure you have Python 3.9 or newer installed. Install required libraries using pip:

```shell
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

### Step 3: Run the Streamlit Application

Navigate to the project directory and run:

```shell
streamlit run streamlit_dashboard.py
```

### Step 4: Access the Dashboard

Open your web browser and navigate to:

```
http://localhost:8501
```

---





