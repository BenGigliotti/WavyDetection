import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

speed_threshold = 1

# load data
df_good_od = pd.read_excel(
    "data/tagHistoryData_09-01-00_09-09-09.xlsx",
    sheet_name="NDC_System_OD_Value"
)
df_good_od['t_stamp'] = pd.to_datetime(df_good_od['t_stamp'])

df_good_speed = pd.read_excel(
    "data/tagHistoryData_09-01-00_09-09-09.xlsx",
    sheet_name="YS_Pullout1_Act_Speed_fpm"
)
df_good_speed['t_stamp'] = pd.to_datetime(df_good_speed['t_stamp'])
df_good_speed = df_good_speed.rename(columns={'Tag_value': 'speed_value'})

df_good_merged = pd.merge(
    df_good_od, 
    df_good_speed[['t_stamp', 'speed_value']], 
    on="t_stamp", 
    how="inner")
df_good_merged = df_good_merged[df_good_merged['speed_value'] > speed_threshold].copy()

df_bad_od = pd.read_excel(
    "data/tagHistoryData_09-24-00_09-26-17 (rejected sample).xlsx",
    sheet_name="NDC_System_OD_Value"
)
df_bad_od['t_stamp'] = pd.to_datetime(df_bad_od['t_stamp'])

df_bad_speed = pd.read_excel(
    "data/tagHistoryData_09-24-00_09-26-17 (rejected sample).xlsx",
    sheet_name="YS_Pullout1_Act_Speed_fpm"
)
df_bad_speed['t_stamp'] = pd.to_datetime(df_bad_speed['t_stamp'])
df_bad_speed = df_bad_speed.rename(columns={'Tag_value': 'speed_value'})

df_bad_merged = pd.merge(
    df_bad_od, 
    df_bad_speed[['t_stamp', 'speed_value']], 
    on="t_stamp", 
    how="inner")
df_bad_merged = df_bad_merged[df_bad_merged['speed_value'] > speed_threshold].copy()


def extract_features(window_data):
    mean_val = np.mean(window_data)
    std_val = np.std(window_data)
    # div by zero
    if mean_val == 0:
        mean_val = 1e-10
    
    # features are normalized to the mean here to take into account spans of multiple different tubing target sizes
    features = {
        'coef_variation': std_val / mean_val,  # coefficient of variation ak relative std
        'relative_range': np.ptp(window_data) / mean_val,  # range relative to mean
        'normalized_variance': np.var(window_data) / (mean_val ** 2),
        'peak_to_peak_ratio': np.ptp(window_data) / np.abs(mean_val),
        'relative_max_deviation': (np.max(window_data) - mean_val) / mean_val,
        'relative_min_deviation': (mean_val - np.min(window_data)) / mean_val,
        # differences between consecutive points, indicates smoothness
        'mean_abs_diff': np.mean(np.abs(np.diff(window_data))) / mean_val,
        'max_abs_diff': np.max(np.abs(np.diff(window_data))) / mean_val,
    }
    
    return features


def create_windows(df, label, window_size):
    features_list = []
    labels_list = []
    
    od_values = df['Tag_value'].values
    num_windows = len(od_values) // window_size
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window = od_values[start_idx:end_idx]
        
        features = extract_features(window)
        features_list.append(features)
        labels_list.append(label)
    
    return features_list, labels_list


def evaluate_models_for_window_size(df_good, df_bad, window_size, n_splits=5):
    good_features, good_labels = create_windows(df_good, label=0, window_size=window_size)
    bad_features, bad_labels = create_windows(df_bad, label=1, window_size=window_size)
    
    print(f"{len(good_features)} good windows and {len(bad_features)} bad windows")
    
    all_features = good_features + bad_features
    all_labels = good_labels + bad_labels
    X = pd.DataFrame(all_features)
    y = np.array(all_labels)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    model_accuracies = {name: [] for name in models.keys()}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            model_accuracies[model_name].append(accuracy)
    
    avg_accuracies = {name: np.mean(accs) for name, accs in model_accuracies.items()} 
    return avg_accuracies


window_sizes = [5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 120]
results = {model: [] for model in ['Logistic Regression', 'Random Forest', 'SVM']}

for window_size in window_sizes:
    print(f"Window size: {window_size} seconds")
    avg_accs = evaluate_models_for_window_size(df_good_merged, df_bad_merged, window_size)
    
    for model_name, accuracy in avg_accs.items():
        results[model_name].append(accuracy)
        print(f"{model_name}: {accuracy:.3f}")
    print()


fig = plt.figure(figsize=(16, 10))

ax1 = plt.subplot(2, 3, 1)
for model_name, accuracies in results.items():
    ax1.plot(window_sizes, accuracies, marker='o', label=model_name, linewidth=2)
ax1.set_xlabel('Window size (sec)', fontsize=11)
ax1.set_ylabel('Average accuracy', fontsize=11)
ax1.set_title('Average model accuracy vs. Window size\n5 split KFolds', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.4, 1.05])

window_size = 30
good_features, good_labels = create_windows(df_good_merged, 0, window_size)
bad_features, bad_labels = create_windows(df_bad_merged, 1, window_size)

X_good = pd.DataFrame(good_features)
X_bad = pd.DataFrame(bad_features)
X_all = pd.concat([X_good, X_bad])
y_all = np.array([0]*len(X_good) + [1]*len(X_bad))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y_all)
y_pred = rf_model.predict(X_scaled)

feature_names = X_all.columns

ax2 = plt.subplot(2, 3, 2)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

ax2.barh(range(len(importances)), importances[indices], align='center')
ax2.set_yticks(range(len(importances)))
ax2.set_yticklabels([feature_names[i] for i in indices])
ax2.set_xlabel('Feature weight', fontsize=11)
ax2.set_title(f'Random Forest feature importance\nWindow size: {window_size} sec', 
             fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

ax3 = plt.subplot(2, 3, 3)
ax3.hist(X_good['mean_abs_diff'], bins=50, alpha=0.6, label='Good', density=True)
ax3.hist(X_bad['mean_abs_diff'], bins=50, alpha=0.6, label='Rejected, chatter', density=True, color='red')
ax3.set_xlabel("Mean Absolute Difference")
ax3.set_ylabel('Density')
ax3.set_title(f'Mean Absolute Difference (smoothness indicator)\nWindow size: {window_size} sec',
              fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

ax4 = plt.subplot(2, 3, 4)
ax4.scatter(X_pca[y_all==0, 0], X_pca[y_all==0, 1], alpha=0.3, s=10, label='Good')
ax4.scatter(X_pca[y_all==1, 0], X_pca[y_all==1, 1], alpha=0.3, s=10, label='Bad (Chatter)', color='red')
ax4.set_xlabel(f'PC1: {pca.explained_variance_ratio_[0]:.1%} variance')
ax4.set_ylabel(f'PC2: {pca.explained_variance_ratio_[1]:.1%} variance')
ax4.set_title('Principle component analysis',
              fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(2, 3, 5)
feature_names = X_all.columns
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

for i, feature in enumerate(feature_names):
    ax5.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
             head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.6)
    ax5.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, 
            fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax5.set_xlabel(f'PC1 weights', fontsize=11)
ax5.set_ylabel(f'PC2 weights', fontsize=11)
ax5.set_title('Feature weights on principle components', 
             fontsize=12, fontweight='bold')
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax5.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax5.grid(True, alpha=0.3)
ax5.axis('equal')

ax6 = plt.subplot(2, 3, 6)
correlation_matrix = X_all.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
           ax=ax6, cbar_kws={'label': 'Correlation'}, square=True)
ax6.set_title('Feature Correlation Matrix', 
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()