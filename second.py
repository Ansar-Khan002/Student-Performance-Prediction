import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/student_combined.csv")

os.makedirs("visualizations", exist_ok=True)
sns.set_style("whitegrid")
count = 0


def save(name):
    global count
    count = count + 1
    path = "visualizations/" + str(count).zfill(2) + "_" + name + ".png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print("saved", path)


plt.figure(figsize=(7, 4))
sns.histplot(data["G3"], bins=20, kde=True)
plt.title("Final Grade Distribution")
plt.xlabel("G3")
save("g3_distribution")

plt.figure(figsize=(7, 4))
sns.boxplot(x="studytime", y="G3", data=data)
plt.title("Study Time vs Final Grade")
save("studytime_vs_g3")

plt.figure(figsize=(7, 4))
sns.boxplot(x="failures", y="G3", data=data)
plt.title("Past Failures vs Final Grade")
save("failures_vs_g3")

plt.figure(figsize=(7, 4))
sns.scatterplot(x="G2", y="G3", data=data, alpha=0.5)
plt.title("Second Term Grade vs Final Grade")
save("g2_vs_g3")

plt.figure(figsize=(7, 4))
sns.boxplot(x="subject", y="G3", data=data)
plt.title("Maths vs Portuguese")
save("subject_vs_g3")

plt.figure(figsize=(11, 9))
sns.heatmap(data.select_dtypes("number").corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
save("correlation_heatmap")

print("Total plots:", count)
