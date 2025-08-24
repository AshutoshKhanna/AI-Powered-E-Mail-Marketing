import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import random
import string
import requests
from io import BytesIO

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

def load_data():
    data = pd.read_csv('walmart_synthetic_data.csv')
    return data

def categorize_customers(data):
    data['Loyal Customer'] = data['Frequency of purchase'] > 10
    return data

def build_dashboard(data):
    st.title("Amazon Customer Segmentation Dashboard")
    
    st.sidebar.header("Filter Options")

    email_target = st.sidebar.selectbox("Select Email Target", 
                                        options=['First Customer', 'First 5 Customers', 'All Customers'],
                                        key='email_target')

    customer_category = st.sidebar.selectbox("Select Customer Category", 
                                             options=['Discount Based', 'Purchase History Based'],
                                             key='customer_category')

    filtered_data = data.copy()
    if customer_category == 'Discount Based':
        filtered_data['Discount Offered'] = np.where(filtered_data['Loyal Customer'], '25%', '15%')
    elif customer_category == 'Purchase History Based':
        filtered_data['Item purchased'] = filtered_data['Item purchased'].astype(str)

    st.subheader("Customer Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(filtered_data))
    col2.metric("Average Purchase", f"${filtered_data['Price'].mean():.2f}")
    col3.metric("Total Revenue", f"${filtered_data['Price'].sum():.2f}")

    st.subheader("Purchase and Ratings Distribution")
    col1, col2 = st.columns(2)

    with col1:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Category', data=filtered_data, palette='coolwarm')
        plt.xticks(rotation=90)
        plt.title('Purchase Distribution by Category')
        st.pyplot(plt)
        plt.close()  

    with col2:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Ratings', data=filtered_data, palette='viridis')
        plt.title('Ratings Distribution')
        st.pyplot(plt)
        plt.close()  

    st.subheader("Customer Age Distribution and Discount Sensitivity Analysis")
    col1, col2 = st.columns(2)

    with col1:
        age_bins = [18, 32, 50, 100]
        age_labels = ['Young', 'Middle-aged', 'Old']
        filtered_data['Age Group'] = pd.cut(filtered_data['Age'], bins=age_bins, labels=age_labels)
        plt.figure(figsize=(6, 4))
        filtered_data['Age Group'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        plt.title('Age Distribution')
        st.pyplot(plt)
        plt.close()

    with col2:
        plt.figure(figsize=(6, 4))
        filtered_data['Discount sensitivity'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Set2'))
        plt.title('Discount Sensitivity Distribution')
        st.pyplot(plt)
        plt.close()

    st.subheader("Top Selling Items")
    top_items = filtered_data.groupby('Item purchased')['Quantity'].sum().reset_index()
    top_items = top_items.sort_values(by='Quantity', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Item purchased', y='Quantity', data=top_items, palette='viridis')
    plt.xticks(rotation=90)
    plt.title('Top Selling Items by Quantity')
    st.pyplot(plt)
    plt.close()

    st.subheader("Sales Trend Analysis")
    filtered_data['Purchased date'] = pd.to_datetime(filtered_data['Purchased date'], format='%d-%m-%Y')
    monthly_sales = filtered_data.resample('M', on='Purchased date')['Price'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales['Purchased date'], monthly_sales['Price'], marker='o', color='b')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales ($)')
    st.pyplot(plt)
    plt.close()

    st.subheader("Customer Segmentation Analysis")
    spending_levels = pd.cut(filtered_data['Price'], bins=[0, 200, 400, np.inf], labels=['Low', 'Mid', 'High'])
    filtered_data['Spending Level'] = spending_levels
    spending_distribution = filtered_data['Spending Level'].value_counts()
    plt.figure(figsize=(6, 4))
    spending_distribution.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('coolwarm'))
    plt.title('Customer Spending Levels')
    st.pyplot(plt)
    plt.close()
    
    return filtered_data, customer_category, email_target

def get_product_link(category, subcategory=None):
    base_url = 'https://www.amazon.in/s?k='
    
    if subcategory:
        return f"{base_url}{category.lower()}+{subcategory.lower().replace(' ', '+')}"
    else:
        return f"{base_url}{category.lower().replace(' ', '+')}"

def generate_discount_code(length=8):
    """Generate a random discount code."""
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def generate_image(prompt):
    """Generate an image using Stability AI's Stable Diffusion model."""
    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STABILITY_API_KEY}",
    }
    
    body = {
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "cfg_scale": 5,
        "samples": 1,
        "text_prompts": [
            {
                "text": prompt,
                "weight": 1
            }
        ],
    }
    
    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code == 200:
        data = response.json()
        image_data = base64.b64decode(data["artifacts"][0]["base64"])
        return BytesIO(image_data)
    else:
        raise Exception(f"Error generating image: {response.text}")

def get_gemini_response(customer_name, customer_category, item_purchased, browsing_history, discount_offered, preferred_brands, product_link, discount_code, prompt):
    input_data = f"""Customer Name: {customer_name}
Customer Category: {customer_category}
Item Purchased: {item_purchased}
Browsing History: {browsing_history}
Discount Offered: {discount_offered}
Preferred Brands: {preferred_brands}
Product Link: {product_link}
Special Discount Code: {discount_code}"""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_data + "\n" + prompt)
    return response.text

def send_email(recipient, subject, body, image_data):
    creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/gmail.send'])
    try:
        service = build('gmail', 'v1', credentials=creds)
        message = MIMEMultipart()
        message['to'] = recipient
        message['subject'] = subject
        
        text_part = MIMEText(body, 'html')
        message.attach(text_part)
        
        image = MIMEImage(image_data.getvalue(), _subtype="png")
        image.add_header('Content-ID', '<image1>')
        message.attach(image)
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        message = service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
        return True
    except HttpError as error:
        st.error(f'An error occurred: {error}')
        return False

def apply_email_marketing_logic(filtered_data, email_target):
    if email_target == 'First Customer':
        recipients = filtered_data.head(1)
    elif email_target == 'First 5 Customers':
        recipients = filtered_data.head(5)
    else:  
        recipients = filtered_data
    
    return recipients

def main():
    data = load_data()
    categorized_data = categorize_customers(data)
    filtered_data, customer_category, email_target = build_dashboard(categorized_data)
    
    if st.sidebar.button("Generate and Send Emails"):
        recipients = apply_email_marketing_logic(filtered_data, email_target)

        total_emails = len(recipients)
        sent_emails = 0
        failed_emails = 0
        
        if customer_category == 'Discount Based':
            input_prompt = (
                "You are a modern-day advertisement specialist working for Amazon E-commerce platform, with expertise in digital marketing and social media trends. "
                "Create a personalized email advertisement for the Amazon customer based on their purchased item and the discount offered. "
                "Remember to keep the tone engaging, modern, and aligned with Walmart's brand image. Your goal is to create an ad that not only catches attention but also drives conversions and sales. "
                "Include the specific discount rate offered to the customer based on their loyalty. "
                "Mention associated brands for the item they purchased, emphasizing Walmart's commitment to quality and value. "
                "Include the provided product link and special discount code in the main ad content. The structure should include:\n"
                "1. A catchy headline (max 10 words)\n"
                "2. Main ad content (must be in 100 words) including the discount rate, associated brands for the purchased item, the product link, and the special discount code\n"
                "3. A powerful call-to-action (5-7 words)\n"
                "4. 5 hashtags for maximizing reach and engagement, including Walmart, preferred brand names, and generic hashtags.\n\n"
                "Format your response as HTML with the following structure:\n"
                "<h1>[Catchy Headline]</h1>\n"
                "<p>[Main Ad Copy]</p>\n"
                "<p><strong>[Call-to-Action]</strong></p>\n"
                "<p>Hashtags: [Hashtag 1], [Hashtag 2], [Hashtag 3], [Hashtag 4], [Hashtag 5]</p>\n"
                "<img src='cid:image1' alt='Product Image' style='max-width: 100%;'>"
            )
        elif customer_category == 'Purchase History Based':
            input_prompt = (
                "You are a modern-day advertisement specialist working for Amazon E-commerce platform, with expertise in digital marketing and social media trends. "
                "Create an email advertisement focusing on the latest products at Amazon related to the customer's browsing history. "
                "The email should highlight new products from popular brands in the categories they have browsed, emphasizing Walmart's wide selection and competitive prices. "
                "Include the provided product link and special discount code in the main ad content. The structure should include:\n"
                "1. A catchy headline (max 10 words)\n"
                "2. Main ad content (must be in 100 words) focusing on new products related to their browsing history, including the product link and special discount code\n"
                "3. A powerful call-to-action (5-7 words)\n"
                "4. 5 hashtags for maximizing reach and engagement, including Walmart, brand names from browsed categories, and generic hashtags.\n\n"
                "Format your response as HTML with the following structure:\n"
                "<h1>[Catchy Headline]</h1>\n"
                "<p>[Main Ad Copy]</p>\n"
                "<p><strong>[Call-to-Action]</strong></p>\n"
                "<p>Hashtags: [Hashtag 1], [Hashtag 2], [Hashtag 3], [Hashtag 4], [Hashtag 5]</p>\n"
                "<img src='cid:image1' alt='Product Image' style='max-width: 100%;'>"
            )
        
        categories = {
            'Electronics': {'TV': ['Samsung', 'LG', 'Sony', 'Vizio'], 'Laptop': ['Dell', 'HP', 'Apple', 'Lenovo'], 'Headphones': ['Bose', 'Sony', 'Beats', 'JBL'], 'Smartphone': ['Apple', 'Samsung', 'Google', 'OnePlus']},
            'Groceries': {'Milk': ['Great Value', 'Horizon', 'Fairlife'], 'Eggs': ['Great Value', 'Eggland\'s Best', 'Vital Farms'], 'Bread': ['Wonder', 'Nature\'s Own', 'Sara Lee'], 'Butter': ['Land O\'Lakes', 'Great Value', 'Kerrygold']},
            'Clothing': {'Shirt': ['Wrangler', 'Levi\'s', 'Hanes', 'Fruit of the Loom'], 'Jeans': ['Wrangler', 'Levi\'s', 'Lee', 'Dickies'], 'Jacket': ['North Face', 'Columbia', 'Carhartt', 'Patagonia'], 'Shoes': ['Nike', 'Adidas', 'Puma', 'Under Armour']},
            'Furniture': {'Table': ['Mainstays', 'Better Homes & Gardens', 'DHP', 'Sauder'], 'Chair': ['Mainstays', 'Better Homes & Gardens', 'DHP', 'Serta'], 'Sofa': ['Mainstays', 'Better Homes & Gardens', 'DHP', 'Serta'], 'Bed': ['Mainstays', 'Better Homes & Gardens', 'DHP', 'Serta']},
            'Toys': {'Lego': ['Lego'], 'Doll': ['Barbie', 'American Girl', 'Disney'], 'Car': ['Hot Wheels', 'Matchbox', 'Disney'], 'Puzzle': ['Ravensburger', 'Melissa & Doug', 'Buffalo Games']},
            'Beauty': {'Skincare': ['Olay', 'Neutrogena', 'CeraVe', 'Aveeno'], 'Makeup': ['Maybelline', 'L\'Oreal', 'CoverGirl', 'Revlon'], 'Hair Care': ['Pantene', 'Garnier', 'Tresemme', 'Herbal Essences'], 'Fragrances': ['Calvin Klein', 'Dolce & Gabbana', 'Versace', 'Chanel']},
            'Appliances': {'Refrigerator': ['Whirlpool', 'LG', 'Samsung', 'GE'], 'Microwave': ['Panasonic', 'Samsung', 'LG', 'GE'], 'Washer': ['Whirlpool', 'LG', 'Samsung', 'GE'], 'Dryer': ['Whirlpool', 'LG', 'Samsung', 'GE']},
            'Health & Wellness': {'Vitamins': ['Nature Made', 'Centrum', 'Vitafusion', 'Nature\'s Bounty'], 'Pain Relief': ['Advil', 'Tylenol', 'Aleve', 'Bayer'], 'First Aid': ['Band-Aid', 'Neosporin', 'Curad', 'Johnson & Johnson'], 'Supplements': ['Nature Made', 'Optimum Nutrition', 'Garden of Life', 'NOW Foods']},
            'Sports & Outdoors': {'Bike': ['Schwinn', 'Huffy', 'Mongoose', 'Trek'], 'Tent': ['Coleman', 'Ozark Trail', 'Wenzel', 'ALPS Mountaineering'], 'Fishing Rod': ['Shakespeare', 'Ugly Stik', 'Abu Garcia', 'Penn'], 'Kayak': ['Lifetime', 'Pelican', 'Intex', 'Old Town']},
            'Home Improvement': {'Drill': ['DeWalt', 'Black+Decker', 'Bosch', 'Makita'], 'Paint': ['Behr', 'Sherwin-Williams', 'Valspar', 'Rust-Oleum'], 'Tool Set': ['Stanley', 'Craftsman', 'Kobalt', 'Husky'], 'Lawn Mower': ['John Deere', 'Husqvarna', 'Toro', 'Honda']}
        }

        
        
        for index, recipient in recipients.iterrows():
            
            browsing_history = eval(recipient['Browsing history'])  
            item_purchased = recipient['Item purchased']
            subject = f"🏃‍♂️ Last Chance: {item_purchased} Flying Off Shelves!Avail the offer now"
            relevant_brands = []
            product_category = ''
            product_subcategory = ''
            
            if customer_category == 'Discount Based':
                for category, subcategories in categories.items():
                    for subcategory, brands in subcategories.items():
                        if subcategory.lower() in item_purchased.lower():
                            relevant_brands.extend(brands)
                            product_category = category
                            product_subcategory = subcategory
                            break
                    if relevant_brands:
                        break
            else:  # Purchase History Based
                for category in browsing_history:
                    if category in categories:
                        product_category = category
                        for subcategory, brands in categories[category].items():
                            relevant_brands.extend(brands)
                        break
            
            relevant_brands = list(set(relevant_brands))
            product_link = get_product_link(product_category, product_subcategory)
            discount_code = generate_discount_code()
            
            body = get_gemini_response(
                customer_name=recipient['Name'],
                customer_category=customer_category,
                item_purchased=item_purchased,
                browsing_history=', '.join(browsing_history),
                discount_offered=recipient.get('Discount Offered', 'N/A'),
                preferred_brands=', '.join(relevant_brands),
                product_link=product_link,
                discount_code=discount_code,
                prompt=input_prompt
            )
            
            # Generate image
            image_prompt = f"Eye-catching modern digital Advertisment for {item_purchased} "
            image_data = generate_image(image_prompt)
            
            if send_email(recipient['Email'], subject, body, image_data):
                sent_emails += 1
            else:
                failed_emails += 1
        
        st.success(f"Emails Sent: {sent_emails}")
        st.error(f"Emails Failed: {failed_emails}")

if __name__ == "__main__":
    main()