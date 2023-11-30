# Zhalgas-Practice13-
import sys
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

geo_data = pd.read_csv("data.csv", header=0, sep=",")

latitude = geo_data["Latitude"]
longitude = geo_data["Longitude"]

# Выполнение линейной регрессии
slope, intercept, r, p, std_err = stats.linregress(latitude, longitude)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, latitude))

# Построение графика
plt.scatter(latitude, longitude)
plt.plot(latitude, mymodel)
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()

# Сохранение графика и вывод в stdout
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv("C:/Users/XE/Downloads/populationkz.csv")

X = data[['latitude', 'longitude']]
y = data['population_2020'] - data['population_2015']

# Добавление константы к переменным для оценки свободного члена (intercept)
X = sm.add_constant(X)

# Создание модели
model = sm.OLS(y, X).fit()

# Вывод таблицы коэффициентов регрессии
coefficients_table = model.summary().tables[1]
print(coefficients_table)

# Визуализация результатов регрессии
plt.scatter(df["Feature Importance"], y, alpha=0.5, label="Actual Values")
plt.plot(df["Feature Importance"], model.predict(X), color='red', label="Regression Line")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.legend()
plt.title("Regression Results")
plt.show()




import pandas as pd
import statsmodels.api as sm

# Загрузка данных
data = pd.read_csv("C:/Users/XE/Downloads/populationkz.csv")

X = data[['latitude', 'longitude']]
y = data['population_2020'] - data['population_2015']

# Добавление константы к переменным для оценки свободного члена (intercept)
X = sm.add_constant(X)

# Создание модели
model = sm.OLS(y, X).fit()

# Вывод результатов
print(model.summary())




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/XE/Downloads/populationkz.csv")

X = data[['latitude', 'longitude']]
y = data['population_2020'] - data['population_2015']

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказания на тестовом наборе
y_pred = model.predict(X_test)

# Оценка качества модели
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))

# Визуализация результатов
plt.scatter(X_test['latitude'], y_test, color='red', label='Actual Values')
plt.plot(X_test['latitude'], y_pred, color='blue', linewidth=3, label='Regression Line')
plt.xlabel("Latitude")
plt.ylabel("Population Change (2020 - 2015)")
plt.title('Linear Regression - Data Science Case')
plt.legend()
plt.show()
