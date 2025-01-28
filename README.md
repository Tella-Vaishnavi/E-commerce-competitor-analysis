# Real-Time Competitor-Strategy-Tracker-for-E-commerce

Project Overview


This project focuses on creating a real-time competitive intelligence tool for e-commerce businesses. It provides actionable insights by monitoring competitor pricing, discount strategies, and customer sentiment. The solution leverages:

� Machine Learning: Predictive modelling with ARIMA.

� LLMs: Sentiment analysis using Hugging Face Transformers and Groq.

� Integration: Slack notifications for real-time�updates.



Features:


* 1.Competitor Data Aggregation: Track pricing and discount strategies.
* 2. Sentiment Analysis: Analyse customer reviews for actionable       insights.
* 3.Predictive modelling: forecast competitor discounts.
* 4.Slack Integration: Get real-time notifications on competitor activity.


Setup Instructions

1.Clone the repository
     
      Git clone <repository-url>
      Cd <repository-directory>

2. Install Dependencies

   Install the required Python libraries using pip:

   pip install -r requirements.txt

3. Configure API Keys

This project requires the following keys:

� Groq API Key: For generating strategic recommendations.

� Slack Webhook URL: For sending�notifications.

Steps:

Groq API Key.

* Sign up for a Groq account at https://groq.com.

* Obtain your API key from the Groq dashboard.

* Use the API key in the app.py file.

Slack Webhook Integration:

o Go to the Slack API.

o Create a new app and enable Incoming Webhooks.

o Add a webhook to a channel and copy the generated URL.

o Add this URL to the�app.py�file.

5. Run the Application

o Run the Streamlit app:

o streamlit�run�app.py

Project Files

o app.py: Main application script.

o scrape.py: Script for web scraping competitor data.

o reviews.csv: Sample reviews data for sentiment analysis.

o competitor_data.csv: Sample competitor data�for�analysis.


Usage

1. Launch the Streamlit app.

2. Select a product from the sidebar.

3. View competitor analysis, sentiment trends, and discount forecasts.

4. Get strategic recommendations and real-time Slack�notifications.


