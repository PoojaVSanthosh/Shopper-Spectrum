# Shopper Spectrum: Customer Segmentation + Product Recommendations

Shopper Spectrum is an end-to-end e-commerce analytics project that combines:

1) **Customer Segmentation** using **RFM analysis + KMeans clustering**  
2) **Product Recommendations** using **Item-based Collaborative Filtering (Cosine Similarity)**  
3) A **Streamlit web app** for real-time interaction (segment prediction + top-N recommendations)

---

## Business Objectives

- Identify meaningful customer segments to support targeted marketing
- Detect at-risk customers for retention strategies
- Provide personalized product recommendations based on purchase history
- Deliver a deployable analytics solution via a simple web application

---

## Dataset

This project uses an Online Retail transactions dataset containing:

- `InvoiceNo`, `StockCode`, `Description`, `Quantity`
- `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

### Important Note (GitHub Size)
The raw dataset (`online_retail.csv`) is **not included** in this repository due to size constraints and best practices.

---

## Methodology Overview

**1. Data Preprocessing**
- Remove missing `CustomerID`
- Remove cancelled invoices (`InvoiceNo` starting with `C`)
- Remove non-positive quantities and unit prices
- Convert dates to datetime
- Create `TotalPrice = Quantity × UnitPrice`

**2. EDA**
- Country-wise transaction patterns
- Top-selling products
- Monthly sales trends
- Monetary distribution (log-scale where required)
- Targeted bivariate analysis

**3. RFM Feature Engineering**
- Recency: days since last purchase
- Frequency: number of unique invoices
- Monetary: total spend

**4. Customer Segmentation**
- Standardize RFM features
- KMeans clustering
- Select `k` using Elbow + Silhouette
- Interpret clusters into business segments:
  - High-Value, Regular, Occasional, At-Risk

**5. Recommendation System**
- Build Customer × Product matrix (implicit feedback via Quantity)
- Item-based cosine similarity between products
- Recommend top 5 similar products

**6. Deployment**
- Streamlit app loads precomputed artifacts for fast inference

---

## Repository Structure

```text
Shopper-Spectrum/
├── artifacts/
│   ├── rfm_scaler.pkl
│   ├── kmeans_model.pkl
│   ├── cluster_labels.json
│   ├── product_list.npy
│   └── similarity_matrix.npy
├── streamlit_app.py
├── requirements.txt
└── README.md
