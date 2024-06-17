
## Project Title

Customer Sentiment Analysis Based on Product Reviews Using Machine Learning: The Case of Cosmetic and Beauty Care Product

## Introduction
This project aims to analyze customer sentiments based on product reviews, focusing on cosmetic and beauty care products. The primary data source is Hasaki's e-commerce website, which sells various cosmetics online. The goal is to create a machine learning model to predict customer sentiments and the aspects they comment on.

## Features
- **Data Collection:** Scraping product reviews from Hasaki's e-commerce website using Selenium.
- **Data Preprocessing:** Standardizing and cleaning data to prepare it for machine learning models.
- **Sentiment Analysis:** Training models to predict customer sentiments (positive, negative, neutral).
- **Aspect Analysis:** Identifying aspects of products (e.g., quality, price) influencing customer reviews.
- **Web Application:** Developing a Streamlit-based web app for real-time sentiment analysis visualization.

## Installation and Usage

### Prerequisites
- Python 3.11
- Selenium
- GitHub
- Streamlit
- Other necessary Python libraries (listed in requirements.txt)

### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/dinhtai507/hasaki_crawling.git
    cd your-repo-name
    ```

2. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

## Directory Structure
    hasaki-sentiment-analysis/
        ├── .devcontainer               
        ├── data
        ├── notebooks/  
        │   ├── preprocesssing.ipynb
        │   └── model.py 
        ├── web
        ├── requirements.txt
        ├── .gitignore
        ├── LICENSE
        ├── README.md      
        └── requirements.txt
### Running the Web Application
1. Run the web scraping script to collect data:
    ```sh
    python notebooks/crawl_comment.ipynb

    ```

2. Preprocess the collected data:
    ```sh
    python notebooks/preprocessing.ipynb
    ```

3. Train the models:
    ```sh
    python notebooks/model.ipynb
    ```

4. Launch the Streamlit web application:
    ```sh
    streamlit run web/app.py
    ```

### Usage
- Access the web app through the provided local URL.
- Standardizing and cleaning data to prepare it for machine learning models.
- Training models to predict customer sentiments (positive, negative, neutral).
- Use the dashboard to visualize customer sentiment and aspect analysis result.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions to improve the project are welcome. Fork the repository, make your changes, and submit a pull request. Please adhere to the project's coding standards and guidelines.

---
Enjoy analyzing customer sentiments and improving product offerings!
