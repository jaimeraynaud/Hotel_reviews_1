import sys
import csv
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

# accepting cookies
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC




path_to_file = "reviewstripadvisor.csv"
num_page = 10
url=  "https://www.tripadvisor.com/Hotel_Review-g188590-d654855-Reviews-Stayokay_Hostel_Amsterdam_Oost-Amsterdam_North_Holland_Province.html"
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(url)
WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='_evidon-accept-button']"))).click()

scrapedReviews=[]

for i in range(0, num_page):

    time.sleep(2)
    container = driver.find_elements_by_xpath("//div[@data-reviewid]")

    for j in range(len(container)):

        rating = container[j].find_element_by_xpath(".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
        review = container[j].find_element_by_xpath(".//div[contains(@class,'pIRBV')]").text.replace("\n", "  ")
        scrapedReviews.append([review,rating]) 
        
    driver.find_element_by_xpath('.//a[@class="ui_button nav next primary "]').click()


scrapedReviewsDF = pd.DataFrame(scrapedReviews, columns=['review', 'rating'])
driver.quit()
print( 'Ready scraping ....')
scrapedReviewsDF.to_csv("reviewtripadvisor.csv", sep=';',index= False)
