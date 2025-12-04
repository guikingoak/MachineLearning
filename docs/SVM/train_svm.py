import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from itertools import cycle

#configurações iniciais
DATA_CSV = "Amazon.csv"   # ajuste se precisar
RANDOM_STATE = 42

#Carregar dados
df = pd.read_csv(DATA_CSV)
print("Raw shape:", df.shape)
print(df["OrderStatus"].value_counts())

# Selecionar colunas
features = [
    "Category","Brand","Quantity","UnitPrice","Discount",
    "Tax","ShippingCost","TotalAmount","PaymentMethod","Country"
]
target = "OrderStatus"

df = df[features + [target]].copy()
df = df.dropna()

#Agrupar marcas
top_brands = df["Brand"].value_counts().nlargest(20).index
df["Brand"] = df["Brand"].where(df["Brand"].isin(top_brands), other="Other")

#Binazirar o Target
df[target] = df[target].astype(str).str.strip().str.lower().map(lambda x: 1 if x=="delivered" else 0)

print("After cleaning:", df.shape)
print("Target counts:", df[target].value_counts().to_dict())

#LabelEncoder para variáveis categóricas
label_encoders = {}
categorical_cols = ["Category","Brand","PaymentMethod","Country"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

#Separar X e y
X = df[features].copy()
y = df[target].copy()

# Escalar as numéricas
numeric_cols = ["Quantity","UnitPrice","Discount","Tax","ShippingCost","TotalAmount"]
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 7) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

#Treinar o SVM
svc = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)
svc.fit(X_train, y_train)

#Avaliação
y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Classification report:\n", report)
print("Confusion matrix:\n", cm)

#Salvar arquivos
pd.DataFrame(classification_report(
    y_test, y_pred, output_dict=True)
).T.to_csv("classification_report.csv")

pd.DataFrame(cm).to_csv("confusion_matrix.csv", index=False)

#Plot da Matriz de Confusão
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Cancelled(0)", "Delivered(1)"],
            yticklabels=["Cancelled(0)", "Delivered(1)"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

#Permutation importance
r = permutation_importance(
    svc, X_test, y_test,
    n_repeats=20,
    random_state=RANDOM_STATE,
    scoring="accuracy"
)

feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": r.importances_mean,
    "importance_std": r.importances_std
}).sort_values(by="importance_mean", ascending=False)

feat_imp.to_csv("permutation_importance.csv", index=False)

print("Top features:")
print(feat_imp.head(10))

#PCA 2D
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_test_pca = pca.fit_transform(X_test)

plt.figure(figsize=(8,6))
palette = {0: "red", 1: "blue"}
for lab in np.unique(y_test):
    idx = (y_test == lab)
    plt.scatter(
        X_test_pca[idx,0], X_test_pca[idx,1],
        label=str(lab),
        c=palette[lab], alpha=0.6, s=40
    )
plt.legend(title="Class (0=Cancelled, 1=Delivered)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Test Data (2D)")
plt.tight_layout()
plt.savefig("pca_scatter.png")
plt.close()

#Curva ROC
y_prob = svc.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
