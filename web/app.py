# Import librabry
import streamlit as st

from datetime import datetime
import numpy as np
from time import sleep
import random
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
import pandas as pd
import os

# App title
st.title('Phân tích cảm xúc từ đánh giá sản phẩm')

# Input link
input_link_button = st.text_input("Input your product link: ")
analyze_button = st.button('Analyze')

# Get rating for comments:
def get_star(string):
    start_index = string.find(':')
    end_index = string.find('%')
    return int(string[start_index+1:end_index]) / 20

# Parse data_product_id
def parse_data_product_id(data_product_id_str):
    # Split string by ","
    id_list = data_product_id_str.split(',')
    # Get unique
    set_list = set()
    set_list.update(id_list)
    # Convert each element in the list to an integer and return
    return [id_ for id_ in set_list]

# ======= GET INFOMATION OF ALL ITEMS
if input_link_button:

    # Declare user agent and argument
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    options = webdriver.ChromeOptions() 
    options.add_argument(f"user-agent={user_agent}")

    # Initialize browser
    driver = webdriver.Chrome(options=options)
    sleep(random.randint(1,5))
    
    # Initialize variable to store data
    name_comment, content_comment, product_variant, datetime_comment, rating_comment = [], [], [], [], []
    
    # Access product page
    driver.get(input_link_button)
    sleep(random.randint(6,7))
    
    # Get data_product_id
    elem_data_productid = driver.find_element(By.CSS_SELECTOR, '.product-add-form input#productId')
    data_product_id = elem_data_productid.get_attribute("value")

    # Get data_product_id_list
    elems_data_productids_list = driver.find_elements(By.CSS_SELECTOR, '.attribute-option-item')
    uniq_data_productids_list = parse_data_product_id(",".join([elem.get_attribute('data-product-ids') for elem in elems_data_productids_list]))
    uniq_data_product_id_str = ",".join(uniq_data_productids_list)
    
    # Get comment_pagination_number
    elems_cmtpage_nums = driver.find_elements(By.CSS_SELECTOR, '.pagination_comment a')
    if elems_cmtpage_nums:
        commentpage_nums = [int(elem.get_attribute('rel')) for elem in elems_cmtpage_nums
                        if elem.get_attribute('rel').isdigit()]
        max_cmtpage = max(commentpage_nums) if commentpage_nums else 1
    else:
        max_cmtpage = 1

    # Get comment details
    for page_num in range(1, 3): #max_cmtpage + 1
        try:
            sleep(random.randint(2,3))
            
            print("Crawl Page " + str(page_num))
            elems_name = driver.find_elements(By.CSS_SELECTOR , ".title_comment strong.txt_color_1")
            name_comment = [elem.text for elem in elems_name] + name_comment
            sleep(random.randint(1,2))

            elems_content = driver.find_elements(By.CSS_SELECTOR , ".item_comment .content_comment")
            content_comment = [elem.text for elem in elems_content] + content_comment
            sleep(random.randint(1,2))

            elems_product_variant = driver.find_elements(By.CSS_SELECTOR , ".item_comment .txt_999")
            product_variant = [elem.text for elem in elems_product_variant] + product_variant
            sleep(random.randint(1,2))

            elems_datetime = driver.find_elements(By.CSS_SELECTOR , ".item_comment .timer_comment")
            datetime_comment = [elem.text for elem in elems_datetime] + datetime_comment
            sleep(random.randint(1,2))

            elems_rating = driver.find_elements(By.CSS_SELECTOR , ".item_comment .number_start")
            rating_comment = [get_star(elem.get_attribute('style')) for elem in elems_rating] + rating_comment
            sleep(random.randint(1,2))
            
            next_pagination_cmt = driver.find_element(By.CSS_SELECTOR, "a.item_next_sort .icon_carret_down")
            actions = ActionChains(driver)
            actions.move_to_element(next_pagination_cmt).click().perform()

            print("Clicked on button next page!")
            sleep(random.randint(2,3))

        except ElementNotInteractableException:
            print("Element Not Interactable Exception!")
            break
        except NoSuchElementException:
            print("Next page button not found or not clickable!")
            break        

    # Add into a dataframe
    comment_data = pd.DataFrame(
        list(zip(name_comment, content_comment, product_variant, datetime_comment, rating_comment)), 
        columns = ['name_comment', 'content_comment','product_variant', 'datetime_comment', 'rating'])
    
    # Add column "link_item", "data_product_id_list", "data_product_id"
    comment_data.insert(0, "link_item", input_link_button)
    comment_data.insert(1, "data_product_id_list", uniq_data_product_id_str)
    comment_data.insert(2, "data_product_id", data_product_id)
    
    # For "data_product_id_list", convert string into list
    comment_data['data_product_id_list'] = comment_data['data_product_id_list'].apply(parse_data_product_id)
    sleep(random.randint(6,7))

    # Print dataframe
    st.write("Dataframe after crawling:")
    st.write(comment_data)
