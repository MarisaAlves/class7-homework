import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

X = diabetes.data
Y = diabetes.target

print(f'sklearn dataset X shape: {X.shape}')
print(f'sklearn dataset y shape: {y.shape}')

print(f'keys: {wine.keys()}')
print(f'data: {wine.data}')
print(f'target: {wine.target}')
print(f'feature_names: {wine.feature_names}')

diabetes_df = pd.DataFrame(data=np.c_[diabetes.data, diabetes.target], columns=diabetes.feature_names + ['target'])
print(diabetes_df.describe())

os.makedirs('Scatter_plots', exist_ok=True)

plt.style.use("ggplot")

for col1_idx, column1 in enumerate(diabetes_df.columns):
    if column1 != "Y" and column1 != "SEX":
        for col2_idx, column2 in enumerate(diabetes_df.columns):
            if col1_idx < col2_idx and column2 != "Y" and column2 != "SEX":
                for col3_idx, column3 in enumerate(diabetes_df.columns):
                    if col1_idx < col3_idx and col2_idx < col3_idx and column3 != "Y" and column3 != "SEX":
                        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
                        axes.grid(axis='y', alpha=0.5)
                        axes.scatter(diabetes_df[column1], diabetes_df["Y"], marker="1", color='blue')
                        axes.scatter(diabetes_df[column2], diabetes_df["Y"], marker="*", color='orange')
                        axes.scatter(diabetes_df[column3], diabetes_df["Y"], marker=".", color='green')
                        axes.set_title(f'Diabetes comparisons')
                        axes.set_ylabel('Diabetes Progression Indicator')
                        axes.set_xlabel('Feature Levels')
                        axes.legend((column1, column2, column3))
                        plt.savefig(f'Scatter_plots/diabetesProgression_to_'+str(column1)+str(column2)+str(column3)+'.png', dpi=300)
                        plt.clf()
                        plt.close()

os.makedirs('other_plots', exist_ok=True)

plt.figure(figsize=(12, 10), dpi=80)
sns.heatmap(diabetes_df.corr(), xticklabels=diabetes_df.corr().columns, yticklabels=diabetes_df.corr().columns, cmap='RdYlGn', center=0, annot=True)
plt.title('Diabetes Heatmap', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f'other_plots/correlogram_diabetes.png', format='png')

diabetes_unaltered = pd.read_csv(filepath_or_buffer='diabetes.data', sep ='\t', header=0)

fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(10, 5))
for indx, col in enumerate(diabetes_unaltered.columns):
    outliers = dict(markerfacecolor='r', marker='D')
    if col != 'SEX':
        if indx == 0:
            axes[0][indx].boxplot(diabetes_unaltered[col], flierprops=outliers)
            axes[0][indx].set_ylabel(col)
        elif indx <= 5:
            axes[0][6-indx].boxplot(diabetes_unaltered[col], flierprops=outliers)
            axes[0][6-indx].set_ylabel(col)
        else:
            axes[1][10-indx].boxplot(diabetes_unaltered[col], flierprops=outliers)
            axes[1][10-indx].set_ylabel(col)

fig.suptitle("Boxplots for Diabetes Features")
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig(f'other_plots/Boxplots_features')
plt.clf()

diabetes_df['SEX'] = diabetes_df['SEX'].astype(int)

fig, axes = plt.subplots(ncols=5, nrows=2)
for idx, column in enumerate(diabetes_unaltered.columns):
    if column != 'SEX':
        if idx == 0:
            sns.violinplot('SEX', column, data=diabetes_unaltered, palette="Set2", ax=axes[0][idx])
        elif idx <= 5:
            sns.violinplot('SEX', column, data=diabetes_unaltered, palette="Set2", ax=axes[0][6 - idx])
        else:
            sns.violinplot('SEX', column, data=diabetes_unaltered, palette="Set2", ax=axes[1][10 - idx])

fig.suptitle("Violin Plot for Features versus Sex")
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig(f'other_plots/ViolinPlots_Sex_to_features.png')
plt.clf()
plt.close()


