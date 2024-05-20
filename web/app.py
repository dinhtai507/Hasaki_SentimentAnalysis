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

import matplotlib.pyplot as plt
import seaborn as sns
from grouped_wordcloud import GroupedColorFunc
from wordcloud import WordCloud

# Page config
st.set_page_config(page_title="Sentiment Dashboard", page_icon=":bar_chart:", layout="wide")

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
@st.cache_data()
def crawl_comments_from_link(input_link_button):

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
    for page_num in range(1, max_cmtpage + 1): # max_cmtpage + 1
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

@st.cache_data()
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

# App title
st.title('Analyzing sentiment from product reviews')

# Input link
input_link_button = st.text_input("Input your product link: ")
analyze_button = st.button('Analyze')
st.write("""
    <style>
        .stApp { 
            width: 100%;
            margin: auto;
            padding: 0 20px; /* Thêm padding để giữ cho nội dung không bị dính vào mép màn hình */
        }
        .stTitle {
            font-size: 20px;
            text-align: center;
        }
        .stTextInput, .stButton { 
            width: 100%; 
            max-width: 800px; /* Giảm kích thước tối đa của input và button để chúng không quá rộng trên màn hình lớn */
            margin-bottom: 10px;
        }
        .stDataFrame {
            width: 100%; 
            max-width: 1600px; /* Tăng kích thước tối đa của dataframe để nó lấp đầy chiều rộng của màn hình */
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

if analyze_button:
    if input_link_button:
        # Crawl data and save into comment_data dataframe
        comment_data = crawl_comments_from_link(input_link_button)

        # Create a list of comments from the selected range
        comment_list = comment_data['content_comment'].tolist()

        # Load trained model
        model_sentiment = joblib.load(os.path.join(os.getcwd(), "notebooks", "svm_model_sentiment.joblib"))
        model_aspect = joblib.load(os.path.join(os.getcwd(), "notebooks", "svm_model_aspect.joblib"))

        # Predict label
        predicted_sentiment, predicted_aspect = predict_output(comment_list)

        # Add to dataframe
        comment_data['predicted_sentiment'] = predicted_sentiment
        comment_data['predicted_aspect'] = predicted_aspect

        # Show dataframe for result
        st.subheader("Analysis Result")
        st.write(comment_data[["name_comment", "content_comment", "predicted_sentiment", "predicted_aspect", "datetime_comment", "product_variant", "rating"]])

        # Set up layout
        col1, col2, col3 = st.columns(3)

        # Biểu đồ 1: Phân phối của nhãn cảm xúc
        sentiment_counts = comment_data['predicted_sentiment'].value_counts()
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'lightblue'}
        colors = [colors[sentiment] for sentiment in sentiment_counts.index] # Tạo mảng màu sắc tương ứng với từng nhãn cảm xúc

        with col1:
            st.subheader("Sentiment Distribution")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
            # ax1.set_title('Distribution of Sentiment Labels')
            ax1.axis('equal')  # Ensures that pie is drawn as a circle
            st.pyplot(fig1)

        # Biểu đồ 2: Tần suất của các nhãn khía cạnh
        aspect_counts = comment_data['predicted_aspect'].value_counts()
        with col2:
            st.subheader("Aspect Distribution")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.pie(aspect_counts, labels=aspect_counts.index, autopct='%1.1f%%')
            # ax2.set_title('Frequency of Aspect Labels')
            ax2.axis('equal')  # Ensures that pie is drawn as a circle
            st.pyplot(fig2)
        
        # Biểu đồ 3: phân phối các cảm xúc theo từng loại nhãn khía cạnh
        with col3:
            st.subheader("Sentiment Distribution by Aspect")
            fig3, ax3 = plt.subplots(figsize=(4,4))
            sns.countplot(x='predicted_aspect', hue='predicted_sentiment', hue_order=['negative', 'neutral', 'positive'], data=comment_data, palette={"neutral": "lightblue", "negative": 'red', "positive": "green"})
            # plt.title('Sentiment Distribution by Aspect Label')
            plt.xlabel('Aspect Label')
            plt.ylabel('Count')
            plt.legend(title='Sentiment Label')
            plt.xticks(rotation=20)
            st.pyplot(fig3)

        # Biểu đồ 4: Word Cloud của các từ khóa trong bình luận
        col4, col5= st.columns(2)
        comment_data['cleaned_Comment'] = preprocessing_text(comment_list)
        all_keywords = ' '.join(comment_data['cleaned_Comment']).split()

        # Tạo danh sách các nhãn cảm xúc cho từng từ
        keyword_sentiments = {}
        for index, row in comment_data.iterrows():
            tokens = row['cleaned_Comment'].split()
            label = row['predicted_sentiment']
            for token in tokens:
                if token not in keyword_sentiments:
                    keyword_sentiments[token] = {'positive': 0, 'negative': 0, 'neutral': 0}
                keyword_sentiments[token][label] += 1

        # Tạo dictionary color_to_words
        color_to_words = {}
        for keyword, sentiments in keyword_sentiments.items():
            max_sentiment = max(sentiments, key=sentiments.get)
            if max_sentiment == 'positive':
                color_to_words.setdefault('#79D70F', []).append(keyword)
            elif max_sentiment == 'negative':
                color_to_words.setdefault('#DA1212', []).append(keyword)
            else:
                color_to_words.setdefault('#A3D8FF', []).append(keyword)        
        # Plot Wordcloud
        with col4:
            # Convert Word Cloud to matplotlib figure
            st.subheader("Word Cloud")
            wc = WordCloud(width=1600, height=1200, background_color='white').generate(' '.join(all_keywords))
            default_color = 'grey'
            grouped_color_func = GroupedColorFunc(color_to_words, default_color)
            wc.recolor(color_func=grouped_color_func)
            fig4, ax4 = plt.subplots(figsize=(4,4))
            ax4.imshow(wc, interpolation="bilinear")
            ax4.axis("off")
            # Display matplotlib figure using st.pyplot()
            st.pyplot(fig4)

        # Biểu đồ 5: Radar chart
        sentiment_aspect_avg = comment_data.groupby('predicted_aspect')['predicted_sentiment'] \
                            .value_counts(normalize=True).unstack()
        sentiment_order = ['positive', 'neutral', 'negative']
        sentiment_aspect_avg = sentiment_aspect_avg.reindex(columns=sentiment_order)

        fig5, ax5 = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
        ax5.set_theta_offset(np.pi/6)
        ax5.set_theta_direction(-1)

        num_sentiments = len(sentiment_aspect_avg.columns)
        angles = np.linspace(0, 2 * np.pi, num_sentiments, endpoint=False).tolist()
        angles.append(angles[0])

        sentiment_names = sentiment_aspect_avg.columns.tolist()
        
        # Plot Radar Chart
        with col5:
            st.subheader("Radar Chart")
            for aspect in sentiment_aspect_avg.index:
                for i, (name, angle) in enumerate(zip(sentiment_names, angles)):
                    ax5.plot([angle, angle], [0, 1], color='black', linewidth=0.25)

                values = sentiment_aspect_avg.loc[aspect].values.tolist()
                values.append(values[0])

                ax5.plot(angles, values, linewidth=1, linestyle='solid', label=aspect)
                ax5.fill(angles, values, alpha=0.1)

            ax5.set_xticks(angles[:-1])
            ax5.set_xticklabels(sentiment_names)

            plt.legend(title='Aspect', loc='lower right')
            # plt.title('Radar Chart of Sentiments for Different Aspects')

            st.pyplot(fig5)

    else:
        st.error("Please input a product link again.")