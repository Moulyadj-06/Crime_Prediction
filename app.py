import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -----------------------
# 1. Load Dataset
# -----------------------
st.title("Crime Analysis & Prediction Dashboard")
data = pd.read_csv(r"C:\Users\csc\Desktop\Crime_BDA_Project\crime_dataset_india.csv")
st.subheader("Raw Data")
st.dataframe(data.head())

# -----------------------
# 2. Preprocessing
# -----------------------
# Fill missing values
for col in data.select_dtypes(include='object').columns:
    data[col].fillna(data[col].mode()[0], inplace=True)
for col in data.select_dtypes(include='number').columns:
    data[col].fillna(data[col].median(), inplace=True)

# Convert dates safely
for col in ['Date of Occurrence', 'Time of Occurrence', 'Date Case Closed']:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors='coerce', dayfirst=True)

# Feature engineering
data['Hour'] = pd.to_datetime(data['Time of Occurrence'], errors='coerce').dt.hour
data['Month'] = data['Date of Occurrence'].dt.month

# Label Encoding for categorical columns
cat_cols = ['City', 'Crime Code', 'Crime Description', 'Victim Gender', 'Weapon Used', 'Crime Domain']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col + '_encoded'] = le.fit_transform(data[col])
    le_dict[col] = le

# Preserve original city names for visualization
city_mapping = dict(zip(data['City_encoded'], data['City']))

# -----------------------
# 3. Sidebar Filters
# -----------------------
st.sidebar.title("Filter Options")

month_options = sorted(data['Month'].dropna().unique())
default_months = month_options.copy()

# City filter
cities_selected = st.sidebar.multiselect(
    "Select Cities", options=data['City'].unique(), default=data['City'].unique()
)
city_codes_selected = [le_dict['City'].transform([c])[0] for c in cities_selected]

# Crime domain filter
crime_domains_selected = st.sidebar.multiselect(
    "Select Crime Domain", options=data['Crime Domain'].unique(), default=data['Crime Domain'].unique()
)
crime_codes_selected = [le_dict['Crime Domain'].transform([c])[0] for c in crime_domains_selected]

# Month filter
months_selected = st.sidebar.multiselect("Select Month", options=month_options, default=default_months)

# Filtered dataset
filtered_data = data[
    (data['City_encoded'].isin(city_codes_selected)) &
    (data['Crime Domain_encoded'].isin(crime_codes_selected)) &
    (data['Month'].isin(months_selected))
].copy()

st.subheader("Filtered Data")
st.dataframe(filtered_data.head())

# -----------------------
# 4. Classification (XGBoost)
# -----------------------
if len(filtered_data) > 0 and len(filtered_data['Crime Domain_encoded'].unique()) > 1:
    X_class = filtered_data[['City_encoded','Crime Code_encoded','Victim Age','Victim Gender_encoded',
                             'Weapon Used_encoded','Hour','Month']]
    y_class = filtered_data['Crime Domain_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    xgb_model = XGBClassifier(eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    st.subheader("XGBoost Classifier Accuracy")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le_dict['Crime Domain'].classes_, columns=le_dict['Crime Domain'].classes_)
    fig, ax = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)
else:
    st.warning("Not enough data to train classifier with current filters.")

# -----------------------
# 5. Regression (HistGradientBoosting)
# -----------------------
if len(filtered_data) > 0:
    X_reg = filtered_data[['City_encoded','Crime Code_encoded','Victim Age','Victim Gender_encoded',
                          'Weapon Used_encoded','Hour','Month']]
    y_reg = filtered_data['Police Deployed']

    scaler = StandardScaler()
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    X_train_r = scaler.fit_transform(X_train_r)
    X_test_r = scaler.transform(X_test_r)

    hgb = HistGradientBoostingRegressor()
    hgb.fit(X_train_r, y_train_r)
    y_pred_r = hgb.predict(X_test_r)

    st.subheader("HistGradientBoosting Regression Results")
    st.write("Mean Squared Error:", mean_squared_error(y_test_r, y_pred_r))
else:
    st.warning("No data available for regression with current filters.")

# -----------------------
# 6. Clustering (KMeans)
# -----------------------
if len(filtered_data) > 0:
    cluster_features = ['City_encoded','Crime Code_encoded','Victim Age','Hour']
    X_cluster = filtered_data[cluster_features]
    X_cluster_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    filtered_data['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

    st.subheader("Crime Clusters (KMeans)")
    fig2 = px.scatter(
        filtered_data, x='Hour', y='Victim Age', color='Cluster',
        hover_data=['City','Crime Description']
    )
    st.plotly_chart(fig2)
else:
    st.warning("No data available for clustering with current filters.")

# -----------------------
# 7. Visualizations
# -----------------------
if len(filtered_data) > 0:
    # Crimes per city
    st.subheader("Crime Trends by City")
    city_counts = filtered_data.groupby('City')['Crime Domain'].count().reset_index()
    fig3 = px.bar(city_counts, x='City', y='Crime Domain', title="Number of Crimes per City")
    st.plotly_chart(fig3)

    # Crimes per hour
    st.subheader("Crime Trends by Hour")
    hour_counts = filtered_data.groupby('Hour')['Crime Domain'].count().reset_index()
    fig4 = px.line(hour_counts, x='Hour', y='Crime Domain', title="Crimes by Hour of Day")
    st.plotly_chart(fig4)

    # -----------------------
    # EXTRA VISUALIZATIONS
    # -----------------------

    # 1. Gender & Weapons
    st.subheader("Victim Gender Distribution")
    fig5 = px.pie(filtered_data, names="Victim Gender", title="Crimes by Victim Gender")
    st.plotly_chart(fig5)

    st.subheader("Weapons Used in Crimes")
    weapon_counts = filtered_data['Weapon Used'].value_counts().reset_index()
    weapon_counts.columns = ["Weapon", "Count"]
    fig6 = px.bar(weapon_counts, x="Weapon", y="Count", title="Weapons Used", color="Count")
    st.plotly_chart(fig6)

    # 2. Yearly Trends
    if 'Date of Occurrence' in filtered_data.columns:
        st.subheader("Crimes Over the Years")
        yearly = filtered_data.groupby(filtered_data['Date of Occurrence'].dt.year)['Crime Domain'].count().reset_index()
        yearly.columns = ['Year', 'Crime Count']
        fig7 = px.line(yearly, x='Year', y='Crime Count', markers=True, title="Yearly Crime Trends")
        st.plotly_chart(fig7)

    # 3. Heatmap Hour vs Day
    if 'Day' not in filtered_data.columns:
        filtered_data['Day'] = filtered_data['Date of Occurrence'].dt.day
    st.subheader("Crime Heatmap (Hour vs Day of Month)")
    heatmap_data = filtered_data.pivot_table(index="Hour", columns="Day", values="Crime Domain", aggfunc="count", fill_value=0)
    fig8, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="Reds", ax=ax)
    st.pyplot(fig8)

    # 4. Top 10 Crime Types
    st.subheader("Top 10 Crime Types")
    top_crimes = filtered_data['Crime Description'].value_counts().head(10).reset_index()
    top_crimes.columns = ["Crime Description", "Count"]
    fig9 = px.bar(top_crimes, x="Crime Description", y="Count", title="Top 10 Crimes", color="Count")
    st.plotly_chart(fig9)

    # 5. Police Deployment vs Crime
    if 'Police Deployed' in filtered_data.columns:
        st.subheader("Police Deployment vs Crime Count")
        deploy_data = filtered_data.groupby('City')['Police Deployed'].sum().reset_index()
        fig10 = px.scatter(deploy_data, x="City", y="Police Deployed", size="Police Deployed", color="City",
                           title="Police Deployment by City")
        st.plotly_chart(fig10)

# # -----------------------
# # 8. Crime Map
# # -----------------------
# if len(filtered_data) > 0:
#     if 'Latitude' in filtered_data.columns and 'Longitude' in filtered_data.columns:
#         st.subheader("Crime Map")
#         map_data = filtered_data.dropna(subset=['Latitude', 'Longitude'])
#         if not map_data.empty:
#             st.map(map_data[['Latitude', 'Longitude']])
#         else:
#             st.warning("No location data available after filtering.")
#     else:
#         st.info("Dataset does not contain Latitude/Longitude columns. Cannot plot map.")
