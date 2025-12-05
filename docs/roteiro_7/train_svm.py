import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

# Configuração inicial
DATA_CSV = "Amazon.csv"
OUT_DIR = "docs"
RANDOM_STATE = 42

def ensure_outdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, name: str):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print("Arquivo salvo:", path)

def save_json(obj, name: str):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("Arquivo salvo:", path)
def main():
    ensure_outdir(OUT_DIR)
    
    
    #carregar dados
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} não encontrado.")
    df = pd.read_csv(DATA_CSV)
    print("Raw shape:", df.shape)
    if "OrderStatus" not in df.columns:
        raise KeyError("Coluna 'OrderStatus' não encontrada.")

    #selecionar colunas
    features = [
        "Category","Brand","Quantity","UnitPrice","Discount",
        "Tax","ShippingCost","TotalAmount","PaymentMethod","Country"
    ]
    target = "OrderStatus"
    df = df[features + [target]].copy()
    df = df.dropna()

    #agrupar marcas menos frequentes
    top_brands = df["Brand"].value_counts().nlargest(20).index
    df["Brand"] = df["Brand"].where(df["Brand"].isin(top_brands), other="Other")

    # mapear target para binário
    df[target] = df[target].astype(str).str.strip().str.lower().map(lambda x: 1 if x == "delivered" else 0)

    print("After cleaning:", df.shape)
    print("Target distribution:", df[target].value_counts().to_dict())

    # LabelEncoder para categóricas
    categorical_cols = ["Category","Brand","PaymentMethod","Country"]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    #separar X e y e escalar numéricas
    X = df[features].copy()
    y = df[target].copy()
    numeric_cols = ["Quantity","UnitPrice","Discount","Tax","ShippingCost","TotalAmount"]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    #treino e test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    #ajustar SVM
    svc = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)
    svc.fit(X_train, y_train)

    #avaliar
    y_pred = svc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("Test accuracy:", acc)

    #salvar resultados numéricos e objetos
    pd.DataFrame(report_dict).T.to_csv(os.path.join(OUT_DIR, "classification_report.csv"))
    pd.DataFrame(cm).to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"), index=False)
    joblib.dump(svc, os.path.join(OUT_DIR, "svc_model.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(label_encoders, os.path.join(OUT_DIR, "label_encoders.joblib"))
    save_json({"after_clean_shape": df.shape, "test_accuracy": float(acc)}, "summary.json")


    # plot da matriz de confusão
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0,1], ["Cancelled(0)", "Delivered(1)"])
    plt.yticks([0,1], ["Cancelled(0)", "Delivered(1)"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
    plt.close()
    print("Saved:", os.path.join(OUT_DIR, "confusion_matrix.png"))

    # Importância por permutação
    r = permutation_importance(svc, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="accuracy")
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values(by="importance_mean", ascending=False)
    feat_imp.to_csv(os.path.join(OUT_DIR, "permutation_importance.csv"), index=False)
    print("Saved:", os.path.join(OUT_DIR, "permutation_importance.csv"))

    # plot PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_test_pca = pca.fit_transform(X_test)
    plt.figure(figsize=(8,6))
    for lab, color in zip([0,1], ["red","blue"]):
        idx = (y_test == lab)
        plt.scatter(X_test_pca[idx,0], X_test_pca[idx,1], label=str(lab), c=color, alpha=0.6, s=40)
    plt.legend(title="Class (0=Cancelled, 1=Delivered)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA - X_test (2D)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_scatter.png"))
    plt.close()
    print("Saved:", os.path.join(OUT_DIR, "pca_scatter.png"))

    # Curva ROC
    y_prob = svc.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
    plt.close()
    print("Saved:", os.path.join(OUT_DIR, "roc_curve.png"))

    #imprimir resumo final
    print("Resumo")
    print("Modelo:", os.path.join(OUT_DIR, "svc_model.joblib"))
    print("Scaler:", os.path.join(OUT_DIR, "scaler.joblib"))
    print("Label encoders", os.path.join(OUT_DIR, "label_encoders.joblib"))
    print("Classification report:", os.path.join(OUT_DIR, "classification_report.csv"))
    print("Matriz de confusão:", os.path.join(OUT_DIR, "confusion_matrix.png"))
    print("Permutation importance:", os.path.join(OUT_DIR, "permutation_importance.csv"))
    print("PCA", os.path.join(OUT_DIR, "pca_scatter.png"))
    print("ROC:", os.path.join(OUT_DIR, "roc_curve.png"))
    print("Summary JSON:", os.path.join(OUT_DIR, "summary.json"))

if __name__ == "__main__":
    main()
