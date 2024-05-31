# EarthQuake-prediction-model

Data Preprocessing: The application preprocesses earthquake data, focusing on Indian regions and extracting relevant features like location, magnitude, and time.

Exploratory Data Analysis (EDA): It displays the distribution of earthquakes among different regions in India and identifies the top earthquake-prone places.

New Model Development: There's a section dedicated to a new predictive model. It preprocesses the data, calculates probabilities based on longitude and latitude occurrences, trains a Random Forest Regressor model, and predicts earthquake probabilities for future years.

Visualization: The app visualizes earthquake-prone places on an interactive map using Pydeck. It also provides a sidebar with details on the top 20 most earthquake-prone locations in India.

User Interaction: Users can explore earthquake data and predictions through buttons and interactive visualizations, enhancing user engagement and understanding.

Libraries used to build the earthquake prediction web application:
Streamlit: For building interactive web applications with Python.
Pandas: For data manipulation and analysis.
Matplotlib: For data visualization, particularly for creating plots and charts.
Scikit-learn (sklearn): For machine learning tasks such as model training, testing, and evaluation.
Pydeck: For creating interactive maps and visualizations.
Geopy: For geocoding and reverse geocoding, used here to retrieve place names based on latitude and longitude coordinates.
