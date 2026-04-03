import numpy as np  # Основная библиотека для вычислений (массивы, матрицы)
import pandas as pd  # Библиотека для работы с таблицами (чтение CSV)
from main import statistical_analysis, plot_histogram, plot_line, plot_heatmap

def main():
    # Загрузка данных
    #path = "data/students_scores.csv"
    path = "data/StudentsPerformance.csv"
    df = pd.read_csv(path)

    # Берем названия всех колонок, где есть слово 'score' или просто все колонки
    # Это позволит коду работать с любым файлом
    target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scores_data = df[target_cols].to_numpy()

    # 1. Статистика по предметам
    stats = statistical_analysis(scores_data)

    # 2. Гистограммы
    for i, col_name in enumerate(target_cols):
        plot_histogram(scores_data[:, i], title=f"Распределение {col_name}")

    # 3. Тепловая карта
    corr_matrix = np.corrcoef(scores_data.T)
    plot_heatmap(corr_matrix, labels=target_cols)

    # 4. Линейный график
    num_students = min(50, scores_data.shape[0])

    plot_line(np.arange(num_students), scores_data[:num_students, :], labels=target_cols)

if __name__ == "__main__":
    main()
