import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, \
    roc_curve, auc
from scipy.stats import norm, chi2_contingency, chi2
import seaborn as sns

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


# 5. Отбор признаков с помощью хи-квадрат
def chi_square_feature_selection(X, y, alpha=0.05):
    selected_features = []

    for col in X.columns:
        try:
            # Для числовых признаков - дискретизируем
            if X[col].dtype in ['int64', 'float64']:
                # Используем квартили для дискретизации
                try:
                    X_disc = pd.qcut(X[col], q=4, duplicates='drop')
                except:
                    X_disc = pd.cut(X[col], bins=4)
            else:
                X_disc = X[col]

            # Создаем таблицу сопряженности
            contingency = pd.crosstab(X_disc, y)

            # Проверяем размеры таблицы
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2_stat, p_val, dof, expected = chi2_contingency(contingency)

                print(f"{col:30s} хи²={chi2_stat:8.2f} p={p_val:8.4f} ", end="")

                if p_val < alpha:
                    print("✓ ЗНАЧИМ")
                    selected_features.append(col)
                else:
                    print("✗ НЕ ЗНАЧИМ")
            else:
                print(f"{col:30s} Недостаточно данных")
        except Exception as e:
            print(f"{col:30s} Ошибка: {str(e)[:50]}...")

    return selected_features


selected_features = chi_square_feature_selection(X, y)
print(f"\nОтобрано признаков: {len(selected_features)} из {X.shape[1]}")
if len(selected_features) > 0:
    X = X[selected_features]
    print(f"Выбранные признаки: {selected_features}")

# 6. Матрица корреляций
plt.figure(figsize=(12, 10))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Матрица корреляции между признаками')
plt.tight_layout()
plt.show()

# 7. Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Разделение на train/test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y.values.reshape(-1, 1),
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val.flatten()
)

# 9. Добавление константы
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_val = np.c_[np.ones(X_val.shape[0]), X_val]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

print(f"\nРазмеры выборок:")
print(f"  Train: {X_train.shape[0]} наблюдений, {X_train.shape[1] - 1} признаков")
print(f"  Val: {X_val.shape[0]} наблюдений")
print(f"  Test: {X_test.shape[0]} наблюдений")


# 10. Функции логистической регрессии
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def LR(X, y, beta):  # логарифм правдоподобия
    z = X @ beta
    ll = y.T @ np.log(sigmoid(z)) + (1 - y).T @ np.log(1 - sigmoid(z))
    return ll[0, 0]


def gradient(X, y, beta):  # Градиент функции правдоподобия
    z = X @ beta
    return X.T @ (y - sigmoid(z))


def gradient_ssr(X, y, beta):  # Градиент для SSR
    z = X @ beta
    p = sigmoid(z)
    # Вычисление градиента SSR
    error = y - p
    p_term = p * (1 - p)
    # Умножаем поэлементно и суммируем
    grad = -2 * X.T @ (error * p_term)
    return grad


# 11. Модель 1: Минимизация отношения правдоподобия (MLE)

def logistic_regression_mle(X, y, lr=0.001, n_iter=5000):
    """Логистическая регрессия с максимизацией правдоподобия"""
    beta = np.zeros((X.shape[1], 1))
    likelihood_history = []

    for i in range(n_iter):
        grad = gradient(X, y, beta)
        beta += lr * grad

        if i % 1000 == 0:
            ll = LR(X, y, beta)
            likelihood_history.append(ll)
            print(f"Log-Likelihood = {ll:.4f}")

    return beta, likelihood_history


beta_mle, likelihood_history = logistic_regression_mle(X_train, y_train)

# 12. Проверка значимости по критерию отношения правдоподобия

def likelihood_ratio_test_all_features(X_train, y_train, beta_full):
    """
    Тест отношения правдоподобия для каждого признака отдельно.
    Убираем один признак и сравниваем с полной моделью.
    """
    # Логарифм правдоподобия полной модели
    ll_full = LR(X_train, y_train, beta_full)
    print(f"Полная модель (все признаки): Log-Likelihood = {ll_full:.4f}")
    print(f"Количество признаков: {X_train.shape[1] - 1}")

    significant_features_indices = []

    # Тестируем каждый признак отдельно
    for feature_idx in range(1, X_train.shape[1]):  # Пропускаем константу (индекс 0)
        # Создаем матрицу без одного признака
        X_reduced = np.delete(X_train, feature_idx, axis=1)

        # Обучаем редуцированную модель
        beta_reduced = np.zeros((X_reduced.shape[1], 1))
        for i in range(1000):
            grad = gradient(X_reduced, y_train, beta_reduced)
            beta_reduced += 0.001 * grad

        ll_reduced = LR(X_reduced, y_train, beta_reduced)

        # Статистика теста
        LR_stat = -2 * (ll_reduced - ll_full)

        # Степени свободы = 1 (один параметр)
        p_value = 1 - chi2.cdf(LR_stat, 1)

        feature_name = selected_features[feature_idx - 1]
        print(f"\nПризнак {feature_idx}: {feature_name}")
        print(f"  Log-Likelihood (без признака): {ll_reduced:.4f}")
        print(f"  Статистика LR: {LR_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            print(f"  Решение: ОСТАВИТЬ (p < 0.05)")
            significant_features_indices.append(feature_idx)
        else:
            print(f"  Решение: УБРАТЬ (p >= 0.05)")

    print(f"\nИтог: оставлено {len(significant_features_indices)} из {X_train.shape[1] - 1} признаков")
    return significant_features_indices


# Выполняем тест отношения правдоподобия
significant_idx = likelihood_ratio_test_all_features(X_train, y_train, beta_mle)

# Оставляем только значимые признаки (включая константу)
significant_idx_with_const = [0] + significant_idx  # Добавляем константу
X_train_lr = X_train[:, significant_idx_with_const]
X_val_lr = X_val[:, significant_idx_with_const]
X_test_lr = X_test[:, significant_idx_with_const]

# Переобучаем модель только на значимых признаках
beta_mle_lr, _ = logistic_regression_mle(X_train_lr, y_train, lr=0.001, n_iter=5000)

# 13. Модель 2: Минимизация SSR
def logistic_regression_ssr(X, y, lr=0.001, n_iter=5000):
    """Логистическая регрессия с минимизацией SSR"""
    beta = np.zeros((X.shape[1], 1))
    ssr_history = []

    for i in range(n_iter):
        # Предсказания
        p = sigmoid(X @ beta)

        # Вычисление SSR
        ssr = np.sum((y - p) ** 2)
        ssr_history.append(ssr)

        # Градиент SSR (минимизируем, поэтому знак минус)
        grad = gradient_ssr(X, y, beta)
        beta -= lr * grad  # Минимизируем SSR

        if i % 1000 == 0:
            print(f"SSR = {ssr:.4f}")

    return beta, ssr_history


# Используем те же значимые признаки для SSR модели
beta_ssr, ssr_history = logistic_regression_ssr(X_train_lr, y_train)

# 14. SSR для обеих моделей
def calculate_ssr_model(X, y, beta):
    """Вычисляет SSR для модели"""
    y_prob = sigmoid(X @ beta)
    ssr = np.sum((y - y_prob) ** 2)
    return ssr


ssr_mle = calculate_ssr_model(X_train_lr, y_train, beta_mle_lr)
ssr_ssr = calculate_ssr_model(X_train_lr, y_train, beta_ssr)

print(f"Модель 1 (MLE): SSR = {ssr_mle:.4f}")
print(f"Модель 2 (SSR минимизация): SSR = {ssr_ssr:.4f}")
print(f"Разница: {abs(ssr_mle - ssr_ssr):.4f}")

# Находим параметры, которые минимизируют SSR
min_ssr_iter = np.argmin(ssr_history)
print(f"Минимальное значение SSR: {ssr_history[min_ssr_iter]:.4f}")

# 15. ROC кривые для обеих моделей

# Предсказания вероятностей для тестовой выборки
y_prob_mle_test = sigmoid(X_test_lr @ beta_mle_lr)
y_prob_ssr_test = sigmoid(X_test_lr @ beta_ssr)

# ROC кривые
fpr_mle, tpr_mle, thresholds_mle = roc_curve(y_test, y_prob_mle_test)
fpr_ssr, tpr_ssr, thresholds_ssr = roc_curve(y_test, y_prob_ssr_test)

roc_auc_mle = auc(fpr_mle, tpr_mle)
roc_auc_ssr = auc(fpr_ssr, tpr_ssr)

print(f"ROC-AUC Модель 1 (MLE): {roc_auc_mle:.4f}")
print(f"ROC-AUC Модель 2 (SSR минимизация): {roc_auc_ssr:.4f}")

# 16. Визуализация ROC кривых
plt.figure(figsize=(12, 10))

# Модель 1 (MLE) - синяя
plt.plot(fpr_mle, tpr_mle, color='blue', lw=3,
         label=f'Модель 1: MLE (AUC = {roc_auc_mle:.3f})')

# Модель 2 (SSR) - красная
plt.plot(fpr_ssr, tpr_ssr, color='red', lw=3, linestyle='--',
         label=f'Модель 2: SSR минимизация (AUC = {roc_auc_ssr:.3f})')

# Случайный классификатор
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':',
         label='Случайный классификатор (AUC = 0.5)')

# Идеальный классификатор
plt.plot([0, 0, 1], [0, 1, 1], color='green', lw=2, linestyle=':',
         alpha=0.5, label='Идеальный классификатор (AUC = 1.0)')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('Сравнение ROC кривых: MLE vs SSR минимизация', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 17. Сравнение качества на тестовых данных
def evaluate_model(y_true, y_prob, model_name):
    """Оценивает модель и выводит метрики"""
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        'Модель': model_name,
        'Общая Точность(Аccuracy)': accuracy_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'Точность положительных предсказаний': precision_score(y_true, y_pred, zero_division=0),
        'Чувствительность': recall_score(y_true, y_pred),
    }

    print(f"\n{model_name}:")
    print(f"  Общая Точность(Аccuracy): {metrics['Общая Точность(Аccuracy)']:.4f}")
    print(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")
    print(f"  Точность положительных предсказаний: {metrics['Точность положительных предсказаний']:.4f}")
    print(f"  Чувствительность: {metrics['Чувствительность']:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    True Negative={cm[0, 0]}  False Positive={cm[0, 1]}")
    print(f"    False Negative={cm[1, 0]}  True Positive={cm[1, 1]}")

    return metrics


metrics_mle = evaluate_model(y_test, y_prob_mle_test, "Модель 1 (MLE)")
metrics_ssr = evaluate_model(y_test, y_prob_ssr_test, "Модель 2 (SSR минимизация)")

# 18. Коэффициенты моделей
print("\n Коэффициенты моделей" )


# Имена значимых признаков (включая константу)
significant_feature_names = ['const'] + [selected_features[i - 1] for i in significant_idx]

# Таблица коэффициентов для MLE
coef_table_mle = pd.DataFrame({
    'Признак': significant_feature_names,
    'Коэффициент_MLE': beta_mle_lr.flatten(),
    'Exp(Коэф)_MLE': np.exp(beta_mle_lr.flatten())
})

# Таблица коэффициентов для SSR модели
coef_table_ssr = pd.DataFrame({
    'Признак': significant_feature_names,
    'Коэффициент_SSR': beta_ssr.flatten(),
    'Exp(Коэф)_SSR': np.exp(beta_ssr.flatten())
})

# Объединяем таблицы
coef_table = pd.merge(coef_table_mle, coef_table_ssr, on='Признак')
print("\nКоэффициенты моделей (только значимые признаки):")
print(coef_table.round(4))

# 19. Сравнение метрик
print("\n Сравнение метрик")
comparison_df = pd.DataFrame([metrics_mle, metrics_ssr])
print(comparison_df.round(4).to_string())

# Разница в метриках
diff_accuracy = abs(metrics_mle['Общая Точность(Аccuracy)'] - metrics_ssr['Общая Точность(Аccuracy)'])
diff_auc = abs(metrics_mle['ROC-AUC'] - metrics_ssr['ROC-AUC'])

print(f"\nРазница в Accuracy: {diff_accuracy:.4f}")
print(f"Разница в ROC-AUC: {diff_auc:.4f}")

if diff_auc < 0.05:
    print("Вывод: Модели имеют схожее качество (разница в AUC < 0.05)")
else:
    if roc_auc_mle > roc_auc_ssr:
        print("Вывод: Модель 1 (MLE) показывает лучшее качество")
    else:
        print("Вывод: Модель 2 (SSR минимизация) показывает лучшее качество")

# 20. Итоговая модель
print("\n Итоговая модель" )

# Выбираем лучшую модель
if roc_auc_mle >= roc_auc_ssr:
    best_beta = beta_mle_lr
    best_model_name = "Модель 1 (MLE)"
    best_auc = roc_auc_mle
else:
    best_beta = beta_ssr
    best_model_name = "Модель 2 (SSR минимизация)"
    best_auc = roc_auc_ssr

print(f"Лучшая модель: {best_model_name}")
print(f"ROC-AUC лучшей модели: {best_auc:.4f}")
print(f"SSR лучшей модели: {calculate_ssr_model(X_test_lr, y_test, best_beta):.4f}")

# Формируем уравнение
equation = f"logit(P) = {best_beta[0, 0]:.4f}"
for i, feature in enumerate(significant_feature_names[1:], 1):
    coef = best_beta[i, 0]
    if coef >= 0:
        equation += f" + {coef:.4f}*{feature}"
    else:
        equation += f" - {abs(coef):.4f}*{feature}"

print(f"\nУравнение лучшей модели:")
print(equation)
print(f"\nВероятность: P(Y=1|X) = 1 / (1 + exp(-({equation.replace('logit(P) = ', '')})))")
print(f"Порог классификации: 0.5")

# 21. График SSR во время обучения (только для SSR модели)
plt.figure(figsize=(10, 6))
plt.plot(ssr_history)
plt.axhline(y=min(ssr_history), color='r', linestyle='--', alpha=0.5, label=f'Min SSR = {min(ssr_history):.2f}')
plt.xlabel('Итерация')
plt.ylabel('SSR')
plt.title('Минимизация SSR во время обучения (Модель 2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 23. Значимость модели

# Логарифм правдоподобия финальной модели
ll_final = LR(X_test_lr, y_test, best_beta)

# Редуцированная модель (только константа) на тестовых данных
X_test_const = X_test_lr[:, 0:1]  # Только константа
beta_const = np.zeros((1, 1))
for i in range(1000):
    p_const = sigmoid(X_test_const @ beta_const)
    grad_const = X_test_const.T @ (y_test - p_const)
    beta_const += 0.001 * grad_const

ll_const = LR(X_test_const, y_test, beta_const)

# Тест отношения правдоподобия
LR_stat_final = -2 * (ll_const - ll_final)
df_final = X_test_lr.shape[1] - 1  # Количество признаков (без константы)
p_value_final = 1 - chi2.cdf(LR_stat_final, df_final)

print(f"Логарифм правдоподобия модели с {df_final} признаками: {ll_final:.4f}")
print(f"Логарифм правдоподобия модели только с константой: {ll_const:.4f}")
print(f"\nСтатистика теста отношения правдоподобия: LR = {LR_stat_final:.4f}")
print(f"Степени свободы: df = {df_final}")
print(f"p-value: {p_value_final:.6f}")

if p_value_final < 0.001:
    significance = "Высоко значима (p < 0.001)"
elif p_value_final < 0.01:
    significance = " очень значима (p < 0.01)"
elif p_value_final < 0.05:
    significance = " значима (p < 0.05)"
else:
    significance = "не значима (p ≥ 0.05)"

print(f"\nМодель {significance}")

if p_value_final < 0.05:
    print(f"Признаки вносят значимый вклад в предсказание диагноза.")
else:
    print(f"Признаки не вносят значимый вклад в предсказание диагноза.")

print("\n" + "="*60)