import pytest
import os
import numpy as np
from main import statistical_analysis, normalize_data


# Фикстура для создания временных данных перед тестами
@pytest.fixture
def sample_data():
    return np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ])


# 1. Тест корректности расчетов статистики
def test_statistics_values(sample_data):
    stats = statistical_analysis(sample_data)

    # Проверяем среднее для первого столбца (10+40+70)/3 = 40
    assert stats['mean'][0] == pytest.approx(40.0)
    # Проверяем максимум для второго столбца
    assert stats['max'][1] == 80
    # Проверяем, что в словаре есть все нужные ключи
    expected_keys = ["mean", "median", "std", "min", "max", "percentile_25", "percentile_75"]
    assert all(key in stats for key in expected_keys)


# 2. Тест нормализации (Min-Max)
def test_normalization_range(sample_data):
    norm = normalize_data(sample_data)
    assert np.min(norm) == 0.0
    assert np.max(norm) == 1.0
    assert norm.shape == sample_data.shape


# 3. Проверка создания файлов графиков
def test_plots_generation():
    # Создаем фиктивные данные для отрисовки
    from main import plot_histogram, plot_heatmap, plot_line

    test_data = np.random.randint(0, 100, (10, 3))

    # Чистим папку plots перед тестом, если она есть
    if os.path.exists("plots/test_hist.png"):
        os.remove("plots/test_hist.png")

    # Запускаем функции
    plot_histogram(test_data[:, 0], title="Test Hist")
    plot_heatmap(np.corrcoef(test_data.T))
    plot_line(np.arange(10), test_data)

    # Проверяем, появились ли файлы на диске
    assert os.path.exists("plots/test_hist.png")
    assert os.path.exists("plots/heatmap.png")
    assert os.path.exists("plots/line_plot.png")


# 4. Тест обработки пустых или некорректных путей
def test_load_dataset_missing_file():
    from main import load_dataset
    # Проверяем, что функция создает файл, если его нет
    path = "data/temp_test.csv"
    if os.path.exists(path):
        os.remove(path)

    data = load_dataset(path)
    assert data.size > 0
    assert os.path.exists(path)
    os.remove(path)