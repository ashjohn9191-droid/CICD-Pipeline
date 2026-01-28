# =======================
# Import Libraries
# =======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score

sns.set(style='white')

# =======================
# Load Dataset
# =======================
dataset = pd.read_csv("iris.csv")

# Clean column names
dataset.columns = [col.strip(" (cm)").replace(" ", "_") for col in dataset.columns]

# =======================
# Feature Engineering
# =======================
dataset["sepal_length_width_ratio"] = dataset["sepal_length"] / dataset["sepal_width"]
dataset["petal_length_width_ratio"] = dataset["petal_length"] / dataset["petal_width"]

dataset = dataset[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "sepal_length_width_ratio",
        "petal_length_width_ratio",
        "target",
    ]
]

# =======================
# Train-Test Split
# =======================
train_data, test_data = train_test_split(
    dataset, test_size=0.2, random_state=44
)

X_train = train_data.drop("target", axis=1).values.astype("float32")
y_train = train_data["target"].values.astype("int32")

X_test = test_data.drop("target", axis=1).values.astype("float32")
y_test = test_data["target"].values.astype("int32")

# =======================
# Logistic Regression
# =======================
logreg = LogisticRegression(
    C=0.0001,
    solver="lbfgs",
    max_iter=100
)

logreg.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)

cm_lr = confusion_matrix(y_test, pred_lr)

f1_lr = f1_score(y_test, pred_lr, average="micro")
prec_lr = precision_score(y_test, pred_lr, average="micro")
recall_lr = recall_score(y_test, pred_lr, average="micro")

train_acc_lr = logreg.score(X_train, y_train) * 100
test_acc_lr = logreg.score(X_test, y_test) * 100

# =======================
# Random Forest
# =======================
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
pred_rf_class = np.round(pred_rf).astype(int)

f1_rf = f1_score(y_test, pred_rf_class, average="micro")
prec_rf = precision_score(y_test, pred_rf_class, average="micro")
recall_rf = recall_score(y_test, pred_rf_class, average="micro")

train_acc_rf = rf.score(X_train, y_train) * 100
test_acc_rf = rf.score(X_test, y_test) * 100

# =======================
# Confusion Matrix Plot
# =======================
def plot_confusion_matrix(cm, class_names):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(10, 7))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm_norm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm_norm[i, j]:.4f}",
            ha="center",
            color="white" if cm_norm[i, j] > thresh else "black",
        )

    plt.ylabel("True Label")
    plt.xlabel(
        f"Predicted Label\naccuracy={accuracy:.4f}; misclass={misclass:.4f}"
    )
    plt.tight_layout()
    plt.savefig("ConfusionMatrix.png", dpi=120)
    plt.close()

plot_confusion_matrix(cm_lr, ["setosa", "versicolor", "virginica"])

# =======================
# Feature Importance Plot
# =======================
importances = rf.feature_importances_
features = dataset.columns[:-1]

fi_df = pd.DataFrame(
    {"feature": features, "importance": importances}
).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=fi_df)
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("FeatureImportance.png", dpi=120)
plt.close()

# =======================
# Save Scores
# =======================
with open("report.md", "w") as f:
    f.write("# Model Training Report\n\n")

    f.write("## Logistic Regression\n")
    f.write(f"- Train Accuracy: {train_acc_lr:.2f}%\n")
    f.write(f"- Test Accuracy: {test_acc_lr:.2f}%\n")
    f.write(f"- F1 Score: {f1_lr:.4f}\n")
    f.write(f"- Precision: {prec_lr:.4f}\n")
    f.write(f"- Recall: {recall_lr:.4f}\n\n")

    f.write("## Random Forest\n")
    f.write(f"- Train Accuracy: {train_acc_rf:.2f}%\n")
    f.write(f"- Test Accuracy: {test_acc_rf:.2f}%\n")
    f.write(f"- F1 Score: {f1_rf:.4f}\n")
    f.write(f"- Precision: {prec_rf:.4f}\n")
    f.write(f"- Recall: {recall_rf:.4f}\n\n")

    f.write("## Confusion Matrix\n")
    f.write("![](ConfusionMatrix.png)\n\n")

    f.write("## Feature Importance\n")
    f.write("![](FeatureImportance.png)\n")


