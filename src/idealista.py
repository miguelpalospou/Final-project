from seleniumwire import webdriver
import pandas as pd
import re
from selenium import webdriver
import undetected_chromedriver as uc
from seleniumwire.undetected_chromedriver import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

def scrapper(APIKEY, PASS):
    options = {
    'proxy': {
        'http': f'http://{APIKEY}:{PASS}@proxy.scrapingbee.com:8886',
    }
}

driver = webdriver.Chrome(seleniumwire_options=options)




l=list()
all_dict={}



for page in range(0,25):
    
    
    if page==1:
        url = "https://www.idealista.com/en/venta-viviendas/barcelona-barcelona/"
    if page>1:
        url = f"https://www.idealista.com/en/venta-viviendas/barcelona-barcelona/pagina-{page}.htm"    
    driver.delete_all_cookies()
    driver.get(url)
    
    time.sleep(5)
    
    html = driver.page_source
    
    
    soup = BeautifulSoup(html, "html.parser")
    All = soup.find_all("div",{"class":"item-info-container"})
    
    time.sleep(4)
   
    for i in range(0,len(All)):
        try:

    
            all_dict["type"]=All[i].find_all("a", {"class":"item-link"})[0].getText().strip().split(' ', 1)[0]
            all_dict["reference"]=All[i].find_all("a", {"class":"item-link"})[0].get('href')
            all_dict["price"]=All[i].find_all("span", {"class":"item-price h2-simulated"})[0].getText()
            all_dict["district"]=All[i].getText().split(",")[0:-1]
            all_dict["street"]=All[i].find("a",{"class":"item-link"}).getText().split(",")[0].split("in")[1].strip()
            all_dict["area"]=All[i].find_all("div", {"class":"item-detail-char"})[0].getText().split('\n')[2]        
            all_dict["rooms"]=All[i].find_all("div", {"class":"item-detail-char"})[0].getText().split('\n')[1] 
            all_dict["plant"]=All[i].find_all("div", {"class":"item-detail-char"})[0].getText().split('\n')[3].split(' ', 1)[0]

            lift_elements = All[i].find_all("div", {"class":"item-detail-char"})
            
            if lift_elements:
                all_dict["lift"]=' '.join(lift_elements[0].getText().split()[7:9])
            else:
                all_dict["lift"]=float('nan')


            parking_elements = All[i].find_all("span", {"class": "item-parking"})
            if parking_elements:
                all_dict["parking"] = parking_elements[0].getText()
            else:
                all_dict["parking"] = float('nan')


            all_dict["description"]=All[i].find("div",{"class":"item-description"}).text.strip("\n")
            print(page)
            l.append(all_dict)
            all_dict={}
        except:
            pass

    data=pd.DataFrame(l)







    





