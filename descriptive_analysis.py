import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def descriptive_analysis(df, numeric_columns, categorical_columns, dataset_name):
    print(f"=== Análise Descritiva: {dataset_name} ===\n")

    if not os.path.exists('plots'):
        os.makedirs('plots')

    if numeric_columns:
        print("Estatísticas para Variáveis Numéricas:\n")
        descriptive_stats = df[numeric_columns].describe().T
        descriptive_stats["mode"] = df[numeric_columns].mode().iloc[0]  # Adiciona moda
        print(descriptive_stats)
        print("\n")
        
        for col in numeric_columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True, bins=30, color="blue")
            plt.title(f"Distribuição da Variável: {col}")
            plt.xlabel(col)
            plt.ylabel("Frequência")
            plt.savefig(f'plots/descriptive_analysis/{dataset_name}_{col}_countplot.png')
            plt.close()
    
    if categorical_columns:
        print("Estatísticas para Variáveis Categóricas:\n")
        for col in categorical_columns:
            print(f"Coluna: {col}")
            print(df[col].value_counts().head(10)) 
            print("\n")
            
            plt.figure(figsize=(10, 6))
            sns.countplot(y=df[col], hue=df[col], palette="viridis", order=df[col].value_counts().index[:10], legend=False)
            plt.title(f"Distribuição de Valores Categóricos: {col}")
            plt.xlabel("Frequência")
            plt.ylabel(col)
            plt.savefig(f'plots/descriptive_analysis/{dataset_name}_{col}_countplot.png')
            plt.close()

files = {
    "customers": "./datasets/olist_customers_dataset.csv",
    "geolocation": "./datasets/olist_geolocation_dataset.csv",
    "order_items": "./datasets/olist_order_items_dataset.csv",
    "order_payments": "./datasets/olist_order_payments_dataset.csv",
    "order_reviews": "./datasets/olist_order_reviews_dataset.csv",
    "orders": "./datasets/olist_orders_dataset.csv",
    "products": "./datasets/olist_products_dataset.csv",
    "sellers": "./datasets/olist_sellers_dataset.csv",
    "category_translation": "./datasets/product_category_name_translation.csv"
}

columns_info = {
    "customers": {"numeric": [], "categorical": ["customer_city", "customer_state"]},
    "geolocation": {"numeric": ["geolocation_lat", "geolocation_lng"], "categorical": ["geolocation_city", "geolocation_state"]},
    "order_items": {"numeric": ["price", "freight_value"], "categorical": ["product_id", "seller_id"]},
    "order_payments": {"numeric": ["payment_value"], "categorical": ["payment_type"]},
    "order_reviews": {"numeric": ["review_score"], "categorical": []},
    "orders": {"numeric": [], "categorical": ["order_status"]},
    "products": {"numeric": ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"], "categorical": ["product_category_name"]},
    "sellers": {"numeric": [], "categorical": ["seller_city", "seller_state"]},
    "category_translation": {"numeric": [], "categorical": ["product_category_name_english"]}
}

for dataset_name, file_path in files.items():
    df = pd.read_csv(file_path)
    numeric_cols = columns_info[dataset_name]["numeric"]
    categorical_cols = columns_info[dataset_name]["categorical"]
    descriptive_analysis(df, numeric_cols, categorical_cols, dataset_name)
