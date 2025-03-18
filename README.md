# Customer Segemntation Using k means Clustering Algorithm

## Overview
This project applies **K-Means Clustering** to segment customers based on their **annual income**, **spending score**, and **age**. The dataset used is the **Mall Customers Dataset**.

## Dataset
The dataset `Mall_Customers.csv` consists of:
- CustomerID (Removed for analysis)
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Methodology
1. **Data Cleaning**: Removed irrelevant columns and handled missing values.
2. **Exploratory Data Analysis (EDA)**:
   - Visualized customer distributions across different attributes.
   - Used histograms, bar plots, and violin plots.
3. **Clustering using K-Means**:
   - Used the **Elbow Method** to find the optimal number of clusters.
   - Performed clustering with 5 clusters.
   - Visualized clusters using **2D and 3D scatter plots**.

## Results
- Customers were segmented into five different clusters.
- Used `seaborn` and `matplotlib` for visualization.
- The 3D scatter plot provides a better understanding of customer segmentation.

