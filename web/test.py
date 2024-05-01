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

import spacy
from deep_translator import GoogleTranslator
from num2words import num2words
import re
import emoji
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, roc_curve, auc

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
def crawl_comments_from_link(link):

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
    for page_num in range(1, 3): # max_cmtpage + 1
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

    return comment_data

# List of stopwords
custom_stopwords = [
        "makeup", "skincare", "beauty", "cosmetics", "lotion", "serum", "moisturizer", "foundation", "lipstick", "mascara",
        "eyeliner", "blush", "concealer", "highlighter", "eyeshadow", "lip", "face", "eyes", "skin", "hair", "brush", "powder",
        "cream", "gel", "toner", "cleanser", "exfoliator", "primer", "sunscreen", "toning", "cleansing", "exfoliating", "hydrating",
        "soothing", "brightening", "anti-aging", "wrinkle", "acne", "pore", "oily", "dry", "sensitive", "combination", "matte",
        "glossy", "shimmer", "natural", "organic", "vegan", "cruelty-free", "fragrance", "scent", "perfume", "cologne", "aroma",
        "essence", "floral", "fruity", "woody", "musk", "vanilla", "jasmine", "rose", "lavender", "citrus", "patchouli",
        "sandalwood", "bergamot", "amber", "oud", "aquatic", "oriental", "gourmand", "spicy", "fresh", "clean", "sweet",
        "floral", "woody", "fruity", "citrusy", "powdery", "green", "herbal", "aromatic", "musky", "sensual", "romantic",
        "exotic", "elegant", "modern", "classic", "sophisticated", "feminine", "masculine", "unisex", "alluring", "captivating",
        "product", "moisturize", "type", "buy", "use",
        "buy", "sell", "online", "commerce", "store", "shop", "purchase", "transaction", "customer", "seller",
        "retail", "e-commerce", "shopping", "sale", "product", "item", "cart", "checkout", "payment", "order",
        "delivery", "shipping", "warehouse", "inventory", "stock", "price", "discount", "deal", "offer", "promotion",
        "coupon", "voucher", "rebate", "refund", "return", "exchange", "warranty", "guarantee", "consumer", "buyer",
        "merchant", "market", "marketplace", "platform", "website", "app", "mobile", "digital", "virtual", "storefront",
        "shopfront", "retail", "shopkeeper", "checkout", "cashier", "payment", "transaction", "shipping", "delivery",
        "order", "basket", "sale", "purchase", "product", "item", "stock", "buying", "selling", "online", "commerce",
        "store", "shop", "purchasing", "customer", "seller", "retail", "e-commerce", "shopping", "sales", "products",
        "items", "carts", "checkout", "payments", "orders", "deliveries", "shipping", "warehouses", "inventories",
        "stocks", "prices", "discounts", "deals", "offers", "promotions", "coupons", "vouchers", "rebates", "refunds",
        "returns", "exchanges", "warranties", "guarantees", "consumers", "buyers", "sellers", "merchants", "markets",
        "marketplaces", "platforms", "websites", "apps", "mobiles", "digitals", "virtuals", "storefronts", "shopfronts",
        "retails", "shopkeepers", "checkouts", "cashiers", "payments", "transactions", "shippings", "deliveries",
        "orders", "baskets", "sales", "purchases", "products", "items", "stocks", "buy", "sell", "online", "commerce",
        "store", "shop", "purchase", "transaction", "customer", "seller", "retail", "e-commerce", "shopping", "sale",
        "product", "item", "cart", "checkout", "payment", "order", "delivery", "shipping", "warehouse", "inventory",
        "stock", "price", "discount", "deal", "offer", "promotion", "coupon", "voucher", "rebate", "refund", "return",
        "exchange", "warranty", "guarantee", "consumer", "buyer", "merchant", "market", "marketplace", "platform",
        "website", "app", "mobile", "digital", "virtual", "storefront", "shopfront", "retail", "shopkeeper", "checkout",
        "cashier", "payment", "transaction", "shipping", "delivery", "order", "basket", "sale", "purchase", "product",
        "item", "stock"
    ]

# Read abbreviation dictionary file
abbs_df = pd.read_csv(os.path.join(os.getcwd(), "data", "abbreviations.csv"))
abbreviation_dict = dict(zip(abbs_df['abbreviation'], abbs_df['meaning']))

# Create Preprocessing function
def preprocessing_text(comments_list):
    cleaned_comments = []  # List to store the cleaned comments

    # Function to decode abbreviations in text
    def decode_abbreviations(text, abbreviation_dict):
        for abbreviation, meaning in abbreviation_dict.items():
            text = re.sub(r'\b' + re.escape(abbreviation) + r'\b', meaning, text)
        return text

    # Function to translate Vietnamese text to English
    def translate_batch_vietnamese_to_english(texts):
        translated_texts = GoogleTranslator(source='vi', target='en').translate_batch(texts)
        return translated_texts

    # Function to convert emojis to text
    def demojize_if_str(text):
        if isinstance(text, str):
            return emoji.demojize(text)
        else:
            return text

    # Function to remove special characters from text
    def remove_special_characters(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Load English language model for lemmatization
    nlp = spacy.load("en_core_web_sm")
    stop_words_spacy = set(nlp.Defaults.stop_words)

    # Combine spaCy stop words with custom stop words
    stop_words_combined = stop_words_spacy.union(custom_stopwords)

    # Function to tokenize text and convert numbers to words
    def tokenize_and_convert_numbers(text):
        tokens = text.split()
        tokens = [num2words(token) if token.isdigit() else token for token in tokens]
        return tokens

    # Function to lemmatize text
    def lemmatize_text(text):
        tokens = tokenize_and_convert_numbers(text)
        tokens = [token.lemma_ for token in nlp(' '.join(tokens)) if isinstance(token, str) or token.text.lower() not in stop_words_combined]
        return tokens
    
    # Iterate through each comment in the list
    for comment in comments_list:
        lower_comment = comment.lower()
        decoded_comment = decode_abbreviations(lower_comment, abbreviation_dict)
        translated_comment = translate_batch_vietnamese_to_english([decoded_comment])[0]
        demojized_comment = demojize_if_str(translated_comment)
        standardized_comment = remove_special_characters(demojized_comment)
        cleaned_comment = lemmatize_text(standardized_comment)
        cleaned_comments.append(' '.join(cleaned_comment))  # Append the cleaned comment to the list

    return cleaned_comments  # Return the list of cleaned comments

# Apply Logistics Regressiong model
def predict_output(text):

    # Preprocess text
    preprocessed_text = preprocessing_text(text)

    # Vectorize preprocess text
    tfidf_vectorizer = joblib.load(os.path.join(os.getcwd(), "notebooks", "tfidf_vectorizer.joblib"))
    transformed_text = tfidf_vectorizer.transform(preprocessed_text)

    # Predict label
    predicted_sentiment = model_sentiment.predict(transformed_text)
    predicted_aspect = model_aspect.predict(transformed_text)

    return predicted_sentiment, predicted_aspect

# Crawl data and save into comment_data dataframe
input_link_button = "https://hasaki.vn/san-pham/nuoc-hoa-hong-khong-mui-klairs-danh-cho-da-nhay-cam-180ml-65994.html"
comment_data = crawl_comments_from_link(input_link_button)

# Create a list of comments from the selected range
comment_list = comment_data['content_comment'].tolist()

# Load trained model
model_sentiment = joblib.load(os.path.join(os.getcwd(), "notebooks", "log_reg_model_sentiment.joblib"))
model_aspect = joblib.load(os.path.join(os.getcwd(), "notebooks", "log_reg_model_aspect.joblib"))

# Predict label
predicted_sentiment, predicted_aspect = predict_output(comment_list)

# Add to dataframe
comment_data['predicted_sentiment'] = predicted_sentiment
comment_data['predicted_aspect'] = predicted_aspect
print(comment_data)