
import sys
import csv
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

path_to_file = "reviewbooking.csv"
num_page = 10
url=  "https://www.booking.com/hotel/nl/stayokay-amsterdam-zeeburg.en-gb.html?aid=397594;label=gog235jc-1DCAEoggI46AdICVgDaKkBiAEBmAEJuAEXyAEM2AED6AEB-AECiAIBqAIDuAKJ8sGKBsACAdICJGRkMjQ1MzMzLTRhMDQtNDlkMS05YzA4LTJlYmZlOGZkMGFhMdgCBOACAQ;sid=d831a6cfa6b79db00ce3f5e0194fcce5;dest_id=-2140479;dest_type=city;dist=0;group_adults=2;group_children=0;hapos=1;hpos=1;no_rooms=1;room1=A%2CA;sb_price_type=total;sr_order=popularity;srepoch=1632663855;srpvid=d4606097e65d006d;type=total;ucfs=1&#tab-reviews"
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(url)

scrapedReviews=[]
driver.find_element_by_xpath('.//button[@id="onetrust-accept-btn-handler"]').click()

for i in range(0, num_page):

    time.sleep(2)
    container = driver.find_elements_by_xpath("//div[contains(@class,'bui-grid__column-9 c-review-block__right')]")

    for j in range(len(container)):

        rating = container[j].find_element_by_xpath(".//span[contains(@class, 'bui-u-sr-only')]").text
        review = container[j].find_element_by_xpath(".//span[contains(@class,'c-review__body')]").text.replace("\n", "  ")
        scrapedReviews.append([review,rating]) 
        
    driver.find_element_by_xpath('.//a[@class="pagenext"]').click()

scrapedReviewsDF = pd.DataFrame(scrapedReviews, columns=['review', 'rating'])
driver.quit()
print( 'Ready scraping ....')
scrapedReviewsDF.to_csv("reviewbooking.csv", sep=';',index= False)
