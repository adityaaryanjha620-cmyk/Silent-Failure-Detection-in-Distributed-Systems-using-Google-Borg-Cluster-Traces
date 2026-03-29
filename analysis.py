# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# =========================
# 2. Load Dataset
# =========================
df = pd.read_csv(r"D:\borg_cluster_project\archive (1)\borg_traces_data.csv")

print("Dataset Loaded Successfully")
print(df.head())
print(df.shape)

# =========================
# 3. Data Cleaning
# =========================
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("\nDataset Info")
print(df.info())

# =========================
# 4. Event Distribution Chart
# =========================
plt.figure(figsize=(8,5))
sns.countplot(x="event", data=df)
plt.title("Cluster Event Distribution")
plt.xticks(rotation=45)
plt.savefig("chart_event_distribution.png")
plt.show()

# =========================
# 5. Failed vs Successful Jobs
# =========================
plt.figure(figsize=(6,4))
sns.countplot(x="failed", data=df)
plt.title("Failure Distribution")
plt.savefig("chart_failure_distribution.png")
plt.show()

# =========================
# 6. Priority Distribution
# =========================
plt.figure(figsize=(8,5))
sns.histplot(df["priority"], bins=30)
plt.title("Priority Distribution")
plt.savefig("chart_priority_distribution.png")
plt.show()

# =========================
# 7. Scheduling Class Distribution
# =========================
plt.figure(figsize=(8,5))
sns.countplot(x="scheduling_class", data=df)
plt.title("Scheduling Class Distribution")
plt.savefig("chart_scheduling_class.png")
plt.show()

# =========================
# 8. Top Machines by Workload
# =========================
machine_load = df["machine_id"].value_counts().head(20)

plt.figure(figsize=(10,6))
machine_load.plot(kind="bar")
plt.title("Top 20 Machines by Workload")
plt.xlabel("Machine ID")
plt.ylabel("Number of Tasks")
plt.savefig("chart_machine_workload.png")
plt.show()

# =========================
# 9. Priority vs Failure (Boxplot)
# =========================
plt.figure(figsize=(6,4))
sns.boxplot(x="failed", y="priority", data=df)
plt.title("Priority vs Failure")
plt.savefig("chart_priority_vs_failure.png")
plt.show()

# =========================
# 10. Instance Index Distribution
# =========================
plt.figure(figsize=(8,5))
sns.histplot(df["instance_index"], bins=50)
plt.title("Instance Index Distribution")
plt.savefig("chart_instance_index.png")
plt.show()

# =========================
# 11. Correlation Heatmap
# =========================
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("chart_heatmap.png")
plt.show()

# =========================
# 12. Event vs Failure Chart
# =========================
pd.crosstab(df["event"], df["failed"]).plot(kind="bar", stacked=True, figsize=(8,5))
plt.title("Event vs Failure")
plt.savefig("chart_event_vs_failure.png")
plt.show()

# =========================
# 13. Feature Engineering
# =========================
df["workload_intensity"] = df["instance_index"] * df["priority"]
df["scheduling_pressure"] = df["priority"] / (df["scheduling_class"] + 1)

# =========================
# 14. Prepare Data for ML
# =========================
features = [
    "time",
    "instance_events_type",
    "scheduling_class",
    "priority",
    "instance_index",
    "workload_intensity",
    "scheduling_pressure"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 15. PCA Visualization
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.5)
plt.title("PCA Visualization of Cluster Workload")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("chart_pca_visualization.png")
plt.show()

# =========================
# 16. Isolation Forest Anomaly Detection
# =========================
model = IsolationForest(contamination=0.03, random_state=42)

df["anomaly"] = model.fit_predict(X_scaled)
df["anomaly"] = df["anomaly"].map({1:0, -1:1})

print("\nAnomaly Counts")
print(df["anomaly"].value_counts())

# =========================
# 17. PCA Anomaly Visualization
# =========================
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df["anomaly"], cmap="coolwarm")
plt.title("Anomaly Detection in Cluster Workload")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("chart_anomaly_detection.png")
plt.show()

print("\nAll Charts Generated Successfully")