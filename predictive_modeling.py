import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

data = pd.read_csv("./customer_data_cluster_customer_segmentation.csv")

data['converted'] = (data['total_spent'] > 100).astype(int)

X = data[['order_count', 'total_spent', 'avg_payment', 'cluster']]
y = data['converted']

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, zero_division=0))
roc_score = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {roc_score:.2f}")

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_score:.2f}")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.legend()
plt.title("Curva ROC")
plt.savefig(f'plots/predictive_modeling/curva_ROC.png')
plt.close()