from selenium import webdriver

import pandas as pd
import numpy as np

import time, datetime, json

YEARS = [2018, 2019, 2020]
MONTHES = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
CITIES = [
            ("Москва (город федерального значения)", "Москва"),
            ("Санкт-Петербург (город федерального значения)", "Санкт-Петербург"),
            ("Новосибирская область", "Новосибирск"),
            ("Свердловская область", "Екатеринбург")
          ]

url_meteopost = "https://www.gismeteo.ru/diary/4079/"

driver = webdriver.Chrome(executable_path = "/home/egoluback/Documents/NTI/2_stage/ML/2_stage/task2/scrapping/chromedriver")
driver.get(url_meteopost)

print("Driver loaded.")

result_arr = {}

for city in CITIES:
    city_result = []
    print("---{0}---".format(city[1]))

    time.sleep(1)

    for yearIndex in range(len(YEARS)):
        driver.find_element_by_xpath("//select[@name = 'Year']/option[text() = '{0}']".format(YEARS[yearIndex])).click()
        for monthIndex in range(len(MONTHES)):
            time.sleep(0.5)
            driver.find_element_by_xpath("//select[@name = 'sd_distr']/option[text() = '{0}']".format(city[0])).click()
            time.sleep(0.5)
            driver.find_element_by_xpath("//select[@name = 'sd_city']/option[text() = '{0}']".format(city[1])).click()

            driver.find_element_by_xpath("//select[@name = 'Month']/option[text() = '{0}']".format(MONTHES[monthIndex])).click()
            driver.find_element_by_xpath("//button[@id = 'selector_go_btn']").click()

            weather_pd = pd.read_html(driver.find_element_by_xpath('//table').get_attribute('outerHTML'))[0]

            temp_day = pd.Series(weather_pd['День']['Температура'], name = "temp_day")
            temp_night = pd.Series(weather_pd['Вечер']['Температура'], name = "temp_night")

            result_temp = pd.merge(temp_day, temp_night, right_index = True, left_index = True)
            city_result.append([YEARS[yearIndex], monthIndex + 1, result_temp.to_dict()])

            print("Progress: {0}%".format((yearIndex * len(MONTHES) + monthIndex + 1) / (len(MONTHES) * len(YEARS)) * 100))

    result_arr[city[1]] = city_result

with open("../data/weather_data.txt", "w+") as file:
    file.write(json.dumps(result_arr))
