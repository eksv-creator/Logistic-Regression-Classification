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

# 5. Отбор признаков с помощью хи - квадрат
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

# 12. Модель 2: Логистическая регрессия
def logistic_regression_standard(X, y, lr=0.001, n_iter=5000):
    """Стандартная логистическая регрессия"""
    beta = np.zeros((X.shape[1], 1))

    for i in range(n_iter):
        # Предсказания
        p = sigmoid(X @ beta)

        # Вычисление ошибки (SSE)
        error = y - p

        # Градиентный спуск
        grad = X.T @ error
        beta += lr * grad

        if i % 1000 == 0:
            sse = np.sum(error ** 2)
            print(f" SSE = {sse:.4f}")

    return beta


beta_standard = logistic_regression_standard(X_train, y_train)

# 13. Тест отношения правдоподобия

def likelihood_ratio_test(beta_full, X_train, y_train):
    """
    Сравнивает полную модель с редуцированной .
    """
    # Логарифм правдоподобия полной модели
    ll_full = LR(X_train, y_train, beta_full)
    print(f"Полная модель: Log-Likelihood = {ll_full:.4f}")

    # Редуцированная модель
    X_const = X_train[:, 0:1]  # Только первый столбец (константа)
    beta_const = np.zeros((1, 1))

    # Обучаем редуцированную модель
    for i in range(1000):
        p_const = sigmoid(X_const @ beta_const)
        grad_const = X_const.T @ (y_train - p_const)
        beta_const += 0.001 * grad_const

    ll_const = LR(X_const, y_train, beta_const)
    print(f"Редуцированная модель: Log-Likelihood = {ll_const:.4f}")

    # Статистика теста
    LR_stat = -2 * (ll_const - ll_full)
    df = X_train.shape[1] - 1  # Разница в количестве параметров

    # p-value
    p_value = 1 - chi2.cdf(LR_stat, df)

    print(f"\nСтатистика теста: LR = {LR_stat:.4f}")
    print(f"Степени свободы: df = {df}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Заключение: полная модель значимо лучше редуцированной (p < 0.05)")
    else:
        print("Заключение: полная модель НЕ значимо лучше редуцированной")

    return LR_stat, p_value


LR_stat, p_value = likelihood_ratio_test(beta_mle, X_train, y_train)

# 14. SSR  для обеих моделей
print("\n" + "=" * 60)
print("Сравнение SSR для обеих моделй")
print("=" * 60)


def calculate_ssr(X, y, beta, threshold=0.5):
    """Вычисляет SSR для модели"""
    y_prob = sigmoid(X @ beta)
    y_pred = (y_prob >= threshold).astype(int)
    ssr = np.sum((y - y_pred) ** 2)
    return ssr


ssr_mle = calculate_ssr(X_train, y_train, beta_mle)
ssr_standard = calculate_ssr(X_train, y_train, beta_standard)

print(f"Модель 1 (MLE): SSR = {ssr_mle:.4f}")
print(f"Модель 2 (Логистическая): SSR = {ssr_standard:.4f}")
print(f"Разница: {abs(ssr_mle - ssr_standard):.4f}")

# 15. ROC КРИВЫЕ для обеих моделей

# Предсказания вероятностей
y_prob_mle_train = sigmoid(X_train @ beta_mle)
y_prob_standard_train = sigmoid(X_train @ beta_standard)

y_prob_mle_val = sigmoid(X_val @ beta_mle)
y_prob_standard_val = sigmoid(X_val @ beta_standard)

y_prob_mle_test = sigmoid(X_test @ beta_mle)
y_prob_standard_test = sigmoid(X_test @ beta_standard)

# ROC кривые для тестовой выборки
fpr_mle, tpr_mle, thresholds_mle = roc_curve(y_test, y_prob_mle_test)
fpr_std, tpr_std, thresholds_std = roc_curve(y_test, y_prob_standard_test)

roc_auc_mle = auc(fpr_mle, tpr_mle)
roc_auc_std = auc(fpr_std, tpr_std)

print(f"ROC-AUC Модель 1 (MLE): {roc_auc_mle:.4f}")
print(f"ROC-AUC Модель 2 (Логистическая): {roc_auc_std:.4f}")

# 16. ВИЗУАЛИЗАЦИЯ ROC КРИВЫХ
plt.figure(figsize=(12, 10))

# Модель 1 (MLE) - синяя
plt.plot(fpr_mle, tpr_mle, color='blue', lw=3,
         label=f'Модель 1: Минимизация правдоподобия (AUC = {roc_auc_mle:.3f})')

# Модель 2 (Стандартная) - красная
plt.plot(fpr_std, tpr_std, color='red', lw=3, linestyle='--',
         label=f'Модель 2: Логистическая (AUC = {roc_auc_std:.3f})')

# Случайный классификатор
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':',
         label='Случайный классификатор (AUC = 0.5)')

# Идеальный классификатор
plt.plot([0, 0, 1], [0, 1, 1], color='green', lw=2, linestyle=':',
         alpha=0.5, label='Идеальный классификатор (AUC = 1.0)')

# Порог 0.5
# Находим ближайший порог к 0.5
thresh_idx_mle = np.argmin(np.abs(thresholds_mle - 0.5))
thresh_idx_std = np.argmin(np.abs(thresholds_std - 0.5))

plt.scatter(fpr_mle[thresh_idx_mle], tpr_mle[thresh_idx_mle],
            color='blue', s=150, marker='o',
            label=f'Порог 0.5 (MLE): FPR={fpr_mle[thresh_idx_mle]:.2f}, TPR={tpr_mle[thresh_idx_mle]:.2f}')

plt.scatter(fpr_std[thresh_idx_std], tpr_std[thresh_idx_std],
            color='red', s=150, marker='s',
            label=f'Порог 0.5 (Логистическая): FPR={fpr_std[thresh_idx_std]:.2f}, TPR={tpr_std[thresh_idx_std]:.2f}')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('Сравнение ROC кривых:\nМинимизация правдоподобия vs Логистическая регрессия', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 17. СРАВНЕНИЕ КАЧЕСТВА НА ТЕСТОВЫХ ДАННЫХ

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

    return metrics


metrics_mle = evaluate_model(y_test, y_prob_mle_test, "Модель 1 (MLE)")
metrics_std = evaluate_model(y_test, y_prob_standard_test, "Модель 2 (Логистическая)")

# 18. КОЭФФИЦИЕНТЫ МОДЕЛЕЙ

# Имена признаков (включая константу)
feature_names = ['const'] + selected_features

# Таблица коэффициентов для MLE
coef_table_mle = pd.DataFrame({
    'Признак': feature_names,
    'Коэффициент_MLE': beta_mle.flatten(),
    'Exp(Коэф)_MLE': np.exp(beta_mle.flatten())
})

# Таблица коэффициентов для стандартной модели
coef_table_std = pd.DataFrame({
    'Признак': feature_names,
    'Коэффициент_Лог': beta_standard.flatten(),
    'Exp(Коэф)_Лог': np.exp(beta_standard.flatten())
})

# Объединяем таблицы
coef_table = pd.merge(coef_table_mle, coef_table_std, on='Признак')
print(coef_table.round(4))

# 19. Сравнение метрик
print("\n Сравнение метрик " )

comparison_df = pd.DataFrame([metrics_mle, metrics_std])
print(comparison_df.round(4))

# Разница в метриках
diff_accuracy = abs(metrics_mle['Общая Точность(Аccuracy)'] - metrics_std['Общая Точность(Аccuracy)'])
diff_auc = abs(metrics_mle['ROC-AUC'] - metrics_std['ROC-AUC'])

print(f"\nРазница в Accuracy: {diff_accuracy:.4f}")
print(f"Разница в ROC-AUC: {diff_auc:.4f}")

if diff_auc < 0.05:
    print("Вывод: Модели имеют схожее качество (разница в AUC < 0.05)")
else:
    if roc_auc_mle > roc_auc_std:
        print("Вывод: Модель 1 (MLE) показывает лучшее качество")
    else:
        print("Вывод: Модель 2 (Лог) показывает лучшее качество")

# 20. Уравнение модели
print("\n Уравнение модели")
if roc_auc_mle >= roc_auc_std:
    best_beta = beta_mle
    best_model_name = "Модель 1 (MLE)"
else:
    best_beta = beta_standard
    best_model_name = "Модель 2 (Лог)"

print(f"Лучшая модель: {best_model_name}")
print(f"ROC-AUC лучшей модели: {max(roc_auc_mle, roc_auc_std):.4f}")

# Формируем уравнение
equation = f"logit(P) = {best_beta[0, 0]:.4f}"
for i, feature in enumerate(feature_names[1:], 1):
    coef = best_beta[i, 0]
    if coef >= 0:
        equation += f" + {coef:.4f}*{feature}"
    else:
        equation += f" - {abs(coef):.4f}*{feature}"

print(f"\nУравнение лучшей модели:")
print(equation)
print(f"\nВероятность: P(Y=1|X) = 1 / (1 + exp(-({equation.replace('logit(P) = ', '')})))")
print(f"Порог классификации: 0.5")

