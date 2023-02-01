# crimes_in_boston
#### Выполнение домашнего задания : " Сборка витрины на Spark "
#### Цель:
Сбор статистики по криминогенной обстановке в разных районах Бостона.  
В качестве исходных данных используется датасет https://www.kaggle.com/AnalyzeBoston/crimes-in-boston  
Программа запускается через spark-submit со следующими аргументами вызова: пути к данным и к результату.  
#### Описание:
Витрина представляет собой агрегат по районам (поле district) со следующими метриками:  
    crimes_total - общее количество преступлений в этом районе;  
    crimes_monthly - медиана числа преступлений в месяц в этом районе;  
    frequent_crime_types - три самых частых crime_type за всю историю наблюдений в этом районе, объединенных через запятую с одним пробелом “, ” , расположенных в порядке убывания частоты, где crime_type - первая часть NAME из таблицы offense_codes, разбитого по разделителю “ - ” (например, если NAME “BURGLARY - COMMERICAL - ATTEMPT”, то crime_type “BURGLARY”);  
    lat - широта координаты района, рассчитанная как среднее по всем широтам инцидентов;  
    lng - долгота координаты района, рассчитанная как среднее по всем долготам инцидентов.  
#### Результат:
 Витрина будет сохранена в один файл в формате .parquet в папке указанной в аргументе вызова.
