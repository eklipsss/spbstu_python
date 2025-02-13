import pandas as pd
import numpy as np

# 1. Привести различные способы создания объектов типа Series
print('-----------------------------------------------------------------------------------')
print('1. Привести различные способы создания объектов типа Series')
# Для создания Series можно использовать
# - списки Python или массивы NumPy
s1_1 = pd.Series([1, 2, 3, 4]) # Через список
s1_2 = pd.Series(np.array([10, 20, 30, 40])) # Через массив NumPy


# - скалярные значение
s1_3 = pd.Series(5, index=['a', 'b', 'c', 'd'])

# - словари
s1_4 = pd.Series({'a': 1, 'b': 2, 'c': 3})
print('\ns1: ')
print(s1_1)
print('\ns2: ')
print(s1_2)
print('\ns3: ')
print(s1_3)
print('\ns4: ')
print(s1_4)

# 2. Привести различные способы создания объектов типа DataFrame
print('\n\n-----------------------------------------------------------------------------------')
print('2. Привести различные способы создания объектов типа DataFrame')
# DataFrame. Способы создания
# - через объекты Series
s2_1 = pd.Series([1, 2, 3], name="col1")
s2_2 = pd.Series([4, 5, 6], name="col2")
df2_1 = pd.DataFrame({'col1': s2_1, 'col2': s2_2})

print('\ns1: ')
print(s2_1)
print('\ns2: ')
print(s2_2)
print('\ndf1: ')
print(df2_1)

# - списки словарей
data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
df2_2 = pd.DataFrame(data)
print('\ndf2: ')
print(df2_2)

# - словари объектов Series
df2_3 = pd.DataFrame({'A': pd.Series([1, 2, 3]), 'B': pd.Series([4, 5, 6])})
print('\ndf3: ')
print(df2_3)

# - двумерный массив NumPy
arr = np.array([[1, 2, 3], [4, 5, 6]])
df2_4 = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print('\ndf4: ')
print(df2_4)

# - структурированный массив Numpy
dtype = [('A', 'int32'), ('B', 'float64')]
values = [(1, 2.5), (3, 4.5)]
arr = np.array(values, dtype=dtype)
df2_5 = pd.DataFrame(arr)
print('\ndf5: ')
print(df2_5)

# 3. Объедините два объекта Series с неодинаковыми множествами ключей (индексов) так, чтобы вместо NaN было установлено значение 1
print('\n\n-----------------------------------------------------------------------------------')
print('3. Объедините два объекта Series с неодинаковыми множествами ключей (индексов) так, чтобы вместо NaN было установлено значение 1')
s3_1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s3_2 = pd.Series([10, 20], index=['b', 'd'])

# Объединение с заменой NaN на 1
s_combined = s3_1.add(s3_2, fill_value=1)

print('\ns1: ')
print(s3_1)
print('\ns2: ')
print(s3_2)
print('\ns_combined: ')
print(s_combined)

# 4. Переписать пример с транслирование для DataFrame так, чтобы вычитание происходило по СТОЛБЦАМ
print('\n\n-----------------------------------------------------------------------------------')
print('4. Переписать пример с транслирование для DataFrame так, чтобы вычитание происходило по СТОЛБЦАМ')
df4_1 = pd.DataFrame({
    'A': [10, 20, 30],
    'B': [40, 50, 60]
})
df4_2 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Вычитание по столбцам
df_diff = df4_1.sub(df4_2)
print()
print(df_diff)

# 5. На примере объектов DataFrame продемонстрируйте использование методов ffill() и bfill()
df = pd.DataFrame({
    'A': [1, np.nan, 3, np.nan, 5],
    'B': [10, 20, np.nan, 40, 50]
})

# ffill() - заполнение пропусков вперед
df_ffill = df.ffill()
print("\nffill:\n", df_ffill)

# bfill() - заполнение пропусков назад
df_bfill = df.bfill()
print("\nbfill:\n", df_bfill)