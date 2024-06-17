
## Project Title

Customer Sentiment Analysis Based on Product Reviews Using Machine Learning: The Case of Cosmetic and Beauty Care Product

## Introduction
This project aims to analyze customer sentiments based on product reviews, focusing on cosmetic and beauty care products. The primary data source is Hasaki's e-commerce website, which sells various cosmetics online. The goal is to create a machine learning model to predict customer sentiments and the aspects they comment on.

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
- Use the dashboard to visualize customer sentiment and aspect analysis.


## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
Enjoy analyzing customer sentiments and improving product offerings!
