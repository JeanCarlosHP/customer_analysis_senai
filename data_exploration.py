import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

dataset_dir = "./datasets"

def load_datasets(directory):
    datasets = {}
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            dataset_name = file.replace(".csv", "")
            datasets[dataset_name] = pd.read_csv(os.path.join(directory, file))
    return datasets

datasets = load_datasets(dataset_dir)

def explore_dataset(df, name):
    print(f"=== Explorando o Dataset: {name} ===")
    print(f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")
    print(f"Colunas: {df.columns.tolist()}")
    print(f"Tipos de Dados:\n{df.dtypes}")
    print(f"Valores Ausentes por Coluna:\n{df.isnull().sum()}")
    print(f"Amostra de Dados:\n{df.head()}")
    print("\n")

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="blue")
        plt.title(f"Distribuição da Variável: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequência")
        plt.show()

    categorical_columns = df.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col].dropna(), palette="viridis", order=df[col].value_counts().index[:10])
        plt.title(f"Distribuição de Valores Categóricos: {col}")
        plt.xlabel("Frequência")
        plt.ylabel(col)
        plt.show()

for name, df in datasets.items():
    explore_dataset(df, name)
