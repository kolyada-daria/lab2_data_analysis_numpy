import os  # Модуль для работы с операционной системой (создание папок, проверка путей)
import numpy as np  # Основная библиотека для вычислений (массивы, матрицы)
import pandas as pd  # Библиотека для работы с таблицами (чтение CSV)
import matplotlib.pyplot as plt  # Для построения графиков
import seaborn as sns  # Для тепловых карт

# Создаем папку для графиков, если её нет
# 'exist_ok=True' предотвращает ошибку, если папка уже есть
os.makedirs("plots", exist_ok=True)


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector():
    """
    Создание массива от 0 до 9.
    Использует np.arange, который работает как встроенный range(), но возвращает ndarray.
    Returns:
        numpy.ndarray: массив чисел от 0 до 9 включительно
    """
    return np.arange(10)


def create_matrix():
    """
    Создание матрицы 5x5 со случайными числами в диапазоне [0,1].
    Returns:
        numpy.ndarray: матрица 5x5 со случайными значениями от 0 до 1
    """
    return np.random.rand(5, 5)


def reshape_vector(vec):
    """
    Преобразование (10,) -> (2,5).
    Изменяет форму массива без изменения его данных.
    Преобразует вектор из 10 элементов в матрицу 2 строки на 5 столбцов.
    Args:
        vec (numpy.ndarray): входной массив формы (10,)
    Returns:
        numpy.ndarray: преобразованный массив формы (2, 5)
    """
    return vec.reshape(2, 5)


def transpose_matrix(mat):
    """
    Транспонирование матрицы.
    Returns:
        numpy.ndarray: транспонированная матрица
    """
    return mat.T


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a, b):
    """
    Сложение векторов одинаковой длины (векторизация без циклов).
    Каждый элемент a[i] складывается с b[i].
    Args:
        a (numpy.ndarray): первый вектор
        b (numpy.ndarray): второй вектор
    Returns:
        numpy.ndarray: результат поэлементного сложения
    """
    return a + b


def scalar_multiply(vec, scalar):
    """
    Умножение вектора на число: каждый элемент массива умножается на скаляр.
    Args:
        vec (numpy.ndarray): входной вектор
        scalar (float/int): число для умножения
    Returns:
        numpy.ndarray: результат умножения вектора на скаляр
    """
    return vec * scalar


def elementwise_multiply(a, b):
    """
    Поэлементное умножение.
    Args:
        a (numpy.ndarray): первый вектор/матрица
        b (numpy.ndarray): второй вектор/матрица
    Returns:
        numpy.ndarray: результат поэлементного умножения
    """
    return a * b


def dot_product(a, b):
    """
    Скалярное произведение.
    Args:
        a (numpy.ndarray): первый вектор
        b (numpy.ndarray): второй вектор
    Returns:
        float: скалярное произведение векторов
    """
    return np.dot(a, b)


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a, b):
    """
    Умножение матриц.
    Args:
        a (numpy.ndarray): первая матрица
        b (numpy.ndarray): вторая матрица
    Returns:
        numpy.ndarray: результат умножения матриц
    """
    return a @ b


def matrix_determinant(a):
    """
    Определитель матрицы.
    Args:
        a (numpy.ndarray): квадратная матрица
    Returns:
        float: определитель матрицы
    """
    return np.linalg.det(a)


def matrix_inverse(a):
    """
    Обратная матрица.
    Args:
        a (numpy.ndarray): квадратная матрица
    Returns:
        numpy.ndarray: обратная матрица
    """
    return np.linalg.inv(a)


def solve_linear_system(a, b):
    """
    Решение системы Ax = b.
    Args:
        a (numpy.ndarray): матрица коэффициентов A
        b (numpy.ndarray): вектор свободных членов b
    Returns:
        numpy.ndarray: решение системы x
    """
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path="data/students_scores.csv"):
    """
    Загрузить CSV и вернуть NumPy массив.
    Использует Pandas для чтения и .to_numpy() для конвертации в массив NumPy.
    Args:
        path (str): путь к CSV файлу
    Returns:
        numpy.ndarray: загруженные данные в виде массива
    """
    if not os.path.exists(path):
        # Если файла нет, создаем его для корректной работы
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(
                "math,physics,informatics\n78,81,90\n85,89,88\n92,94,95\n70,75,72\n88,84,91\n95,99,98\n60,65,70\n73,70,68\n84,86,85\n90,93,92")

    df = pd.read_csv(path)
    return df.to_numpy()


def statistical_analysis(data):
    """
    Словарь со статистическими показателями.
    Вычисляет основные статистические метрики для набора данных.
    Нужно оценить:
        - средний балл
        - медиану
        - стандартное отклонение
        - минимум
        - максимум
        - 25 и 75 перцентили
    Args:
        data (numpy.ndarray): одномерный массив данных
    Returns:
        dict: словарь со статистическими показателями
    """
    return {
        "mean": np.mean(data),                         # Среднее арифметическое (средний балл)
        "median": np.median(data),                     # Медиана (середина отсортированного списка)
        "std": np.std(data),                           # Стандартное отклонение (разброс данных)
        "min": np.min(data),                           # Минимальное значение
        "max": np.max(data),                           # Максимальное значение
        "percentile_25": np.percentile(data, 25),   # 25% результатов ниже этого значения
        "percentile_75": np.percentile(data, 75)    # 75% результатов ниже этого значения
    }


def normalize_data(data):
    """
    Min-Max нормализация.
    Формула: (x - min) / (max - min)
    Args:
        data (numpy.ndarray): входной массив данных
    Returns:
        numpy.ndarray: нормализованный массив данных в диапазоне [0, 1]
    """
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data):
    """
    Гистограмма распределения оценок.
    Args:
        data (numpy.ndarray): данные для гистограммы
    """
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=10, color='skyblue', edgecolor='black')
    plt.title("Распределение оценок по математике")
    plt.xlabel("Баллы")
    plt.ylabel("Количество студентов")
    plt.savefig("plots/histogram.png") # Сохранение в файл
    plt.close() # Закрываем график, чтобы освободить оперативную память


def plot_heatmap(matrix):
    """
    Тепловая карта корреляции.
    Чем ярче цвет, тем сильнее связь между переменными.
    Args:
        matrix (numpy.ndarray): матрица корреляции
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap='coolwarm')
    plt.title("Тепловая карта корреляции предметов")
    plt.savefig("plots/heatmap.png")
    plt.close()


def plot_line(x, y):
    """
    График зависимости: студент -> оценка.
    Args:
        x (numpy.ndarray): номера студентов
        y (numpy.ndarray): оценки студентов
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', linestyle='-', color='green')
    plt.title("Оценки студентов по математике")
    plt.ylabel("Балл")
    plt.grid(True)
    plt.savefig("plots/line_plot.png")
    plt.close()

if __name__ == "__main__":
    print("Запустите python3 -m pytest test.py -v для проверки лабораторной работы.")