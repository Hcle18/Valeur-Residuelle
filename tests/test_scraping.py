import pandas as pd
import time
import re
from datetime import datetime
import sys
import os
import glob
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC




model_test = "Fiat 500 X"
option_web_1 = "KA+"
option_web_2 = "500"

print((model_test in option_web_1) or (option_web_1 in model_test))
print((model_test in option_web_2) or (option_web_2 in model_test))

option_web_1_cleaned = re.sub(r"[\s\-\']", "", option_web_1)
print(f'Option web 1 cleaned is: {option_web_1_cleaned}')



# Initialize the Edge driver
service = Service(EdgeChromiumDriverManager().install())
options = webdriver.EdgeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")  # Cache Selenium
#options.add_argument('--headless')  # Exécute le navigateur en arrière-plan
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")  # Facultatif : exécute en arrière-plan

driver = webdriver.Edge(service=service, options=options)
driver.maximize_window()  # Maximiser la fenêtre du navigateur

print(driver.capabilities['browserVersion'])  # version du navigateur Edge
print(driver.capabilities['msedge']['msedgedriverVersion'])  # version du driver Edge

driver.quit()
    
