import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

# 1. Загрузка данных
df = pd.read_excel("summary_01_12_18_bez_fio.xlsx")
print("Размер данных:", df.shape)
print("Названия столбцов:", df.columns.tolist())

# 2. Целевая переменная
y = df['pnp_diagnosis']
mask = y.notna()
y = y.loc[mask]

# 3. Признаки
X = df.drop(columns=['pnp_diagnosis'])
if 'id' in X.columns:
    X = X.drop(columns=['id'])
X = X.loc[mask]

# 4. Обработка пропусков
X = X.fillna(X.median())

# 5. Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Разделение на train/validation/test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y.values,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,  # 0.25 * 0.8 = 0.2 от общего объема
    random_state=42,
    stratify=y_train_val
)

# 7. Добавление константы
X_train_sm = sm.add_constant(X_train)
X_val_sm = sm.add_constant(X_val)
X_test_sm = sm.add_constant(X_test)

print(f"\nРазмеры выборок:")
print(f"  Train: {X_train_sm.shape[0]} наблюдений")
print(f"  Val: {X_val_sm.shape[0]} наблюдений")
print(f"  Test: {X_test_sm.shape[0]} наблюдений")

# 8. Логистическая регрессия (MLE)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit(disp=False)

print("\nРЕЗУЛЬТАТЫ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ:")
print(result.summary())

# 9. Таблица коэффициентов
coef_names = ['const'] + df.drop(columns=['pnp_diagnosis','id']).columns.tolist()

coef_series = pd.Series(result.params, index=coef_names)
se_series = pd.Series(result.bse, index=coef_names)
z_series = pd.Series(result.tvalues, index=coef_names)
p_series = pd.Series(result.pvalues, index=coef_names)

coef_table = pd.DataFrame({
    'Признак': coef_series.index,
    'Коэффициент': coef_series.values,
    'СтОшибка': se_series.values,
    'Z-стат': z_series.values,
    'p_знач': p_series.values,
    'ОШ': np.exp(coef_series.values)
})

# Добавляем столбец "Значим"
coef_table['Значим'] = np.where(coef_table['p_знач'] < 0.05, 'Да', 'Нет')

print("\nКОЭФФИЦИЕНТЫ МОДЕЛИ:")
print(coef_table.round(4))

#  10. Подбор оптимального порога на validation
y_prob_val = result.predict(X_val_sm)
thresholds = np.linspace(0.1, 0.9, 50)
best_threshold = 0.5
best_f1 = 0

for thresh in thresholds:
    y_pred_val_thresh = (y_prob_val >= thresh).astype(int)
    f1 = f1_score(y_val, y_pred_val_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\nЛучший порог: {best_threshold:.3f} (F1 = {best_f1:.3f})")

# 11. Прогноз на тестовых данных
y_prob_test = result.predict(X_test_sm)
y_pred_test = (y_prob_test >= best_threshold).astype(int)

#  12. Качество на тестовых данных
print("\nКАЧЕСТВО НА ТЕСТОВЫХ ДАННЫХ:")
print(f"Порог отсечения: {best_threshold:.3f}")
print(f"Точность: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")
print(f"Чувствительность: {recall_score(y_test, y_pred_test):.4f}")

cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(f"TN={cm[0,0]}  FP={cm[0,1]}")
print(f"FN={cm[1,0]}  TP={cm[1,1]}")

# 13. SSR и WSSR
y_pred_train = (result.predict(X_train_sm) >= best_threshold).astype(int)
SSR = np.sum((y_train - y_pred_train) ** 2)
WSSR = np.sum(result.predict(X_train_sm) * (1 - result.predict(X_train_sm)) * (y_train - result.predict(X_train_sm))**2)
print(f"\nSSR = {SSR:.4f}")
print(f"WSSR = {WSSR:.4f}")

# 14. LRI (McFadden)
ll_model = result.llf
ll_null = result.llnull
LRI = 1 - (ll_model / ll_null)
print(f"LRI (McFadden) = {LRI:.4f}")

# 15. Визуализация
coef_df = coef_table[coef_table['Признак'] != 'const'].copy()
feature_labels = {name: name for name in coef_df['Признак']}
coef_df['label'] = coef_df['Признак'].map(feature_labels)
coef_df = coef_df.sort_values(by='Коэффициент')

plt.figure(figsize=(12, 14))
y_pos = range(len(coef_df))
for i, row in coef_df.iterrows():
    color = 'red' if row['p_знач'] < 0.05 else 'grey'
    plt.barh(i, row['Коэффициент'], color=color)

plt.yticks(y_pos, coef_df['label'])
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Влияние признаков на вероятность диагноза (логистическая регрессия)")
plt.xlabel("Коэффициент (логит)")
plt.ylabel("Признак")
plt.tight_layout()
plt.show()

# 16. Уравнение логит-модели для значимых признаков
significant_features = coef_table[coef_table['p_знач'] < 0.05]
if len(significant_features) > 0:
    print("\nУРАВНЕНИЕ ЛОГИТ-МОДЕЛИ:")
    equation = f"logit(P) = {result.params[0]:.4f}"
    for idx, row in significant_features[significant_features['Признак'] != 'const'].iterrows():
        feature_idx = coef_names.index(row['Признак'])
        equation += f" + ({result.params[feature_idx]:.4f})*{row['Признак']}"
    print(equation)
else:
    print("\nНет статистически значимых признаков для уравнения.")
