import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("./datasets/olist_customers_dataset.csv")
orders = pd.read_csv("./datasets/olist_orders_dataset.csv")
order_items = pd.read_csv("./datasets/olist_order_items_dataset.csv")
payments = pd.read_csv("./datasets/olist_order_payments_dataset.csv")

merged = orders.merge(order_items, on="order_id").merge(payments, on="order_id").merge(customers, on="customer_id")

print(merged.head())

customer_data = merged.groupby("customer_unique_id").agg({
    "price": "sum",
    "order_id": "count",
    "payment_value": "mean"
}).rename(columns={
    "price": "total_spent",
    "order_id": "order_count",
    "payment_value": "avg_payment"
}).reset_index()

print()
print(customer_data.head())

scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[["total_spent", "order_count", "avg_payment"]])

print()
print(customer_data_scaled[:5])

inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customer_data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertia, marker='o')
plt.title("Método do Cotovelo")
plt.xlabel("Número de Clusters")
plt.ylabel("Inércia")
plt.savefig(f'plots/customer_segmentation/cotovelo_plot.png')
plt.close()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_data["cluster"] = kmeans.fit_predict(customer_data_scaled)

customer_data.to_csv("customer_data_cluster_customer_segmentation.csv", index=False)
print(customer_data["cluster"])

numeric_cols = customer_data.select_dtypes(include=["float64", "int64"]).columns
print()
print("numeric_cols:", numeric_cols)
print()
print(customer_data.dtypes)
print()
print(customer_data.groupby("cluster")[numeric_cols].mean())

sns.pairplot(customer_data, hue="cluster", vars=["total_spent", "order_count", "avg_payment"])
plt.savefig(f'plots/customer_segmentation/kmeans_plot.png')
plt.close()
