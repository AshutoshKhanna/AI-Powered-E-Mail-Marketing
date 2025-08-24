# E-commerce Customer Segmentation and AI-Powered Marketing Dashboard

This project is a Streamlit application that provides a comprehensive dashboard for analyzing customer data and automates a personalized email marketing campaign. It leverages AI models from **Google** and **Stability AI** to create dynamic and targeted advertisements for customers based on their purchase behavior.

![Dashboard Screenshot](/dashboard.png)
---

## Key Features 

* **Interactive Dashboard**: Visualizes customer data, including purchase distribution, sales trends, age demographics, and spending levels, to provide a detailed overview of the customer base.
* **Customer Segmentation**: Categorizes customers based on their purchase frequency and loyalty. The dashboard allows for targeted marketing strategies based on these segments.
* **Personalized AI-Generated Content**: Utilizes Google's **Gemini API** to generate tailored, brand-specific email ad copy for each customer.
* **AI Image Generation**: Creates custom, visually appealing product images for the emails using Stability AI's **Stable Diffusion** model.
* **Automated Email Sending**: Integrates with the **Gmail API** to send out personalized email campaigns to selected customer groups.

---

## Prerequisites 

Before running the application, ensure you have the following installed and configured:

* **Python 3.11*
* **API Keys**:

  * A Google API Key for the **Gemini Pro** model.
  * A Stability AI API Key for image generation.
* **Gmail API Credentials**:

  * A Google account with a Google Cloud project.
  * The Gmail API enabled in your Google Cloud project.
  * An OAuth 2.0 Client ID for a desktop app.
  * The `token.json` file generated after first running a Gmail API script.

---

## Installation and Setup 

1. **Clone the repository:**

```bash
git clone <repository_url>
cd <repository_folder>
```

2. **Install the required libraries:**

```bash
pip install -r requirements.txt
```

3. **Set up API Keys:**

Create a `.env` file in the root directory and add:

```env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
STABILITY_API_KEY="YOUR_STABILITY_AI_API_KEY"
```

4. **Configure Gmail API:**

   * Follow the [official Google documentation](https://developers.google.com/gmail/api/quickstart/python) to set up Gmail API.
   * Place the `token.json` file in the project's root directory.

5. **Data File:**

   * Ensure `walmart_synthetic_data.csv` is in the root folder. This file contains synthetic customer data.

---

## How to Use 

1. **Run the Streamlit application:**

```bash
streamlit run your_script_name.py
```

2. **Navigate the Dashboard:**

   * Opens in your default browser.
   * Displays various visualizations of customer data.
   * Sidebar lets you configure your email campaign.

3. **Configure the Email Campaign:**

   * **Select Email Target:** First Customer, First 5 Customers, or All Customers.
   * **Select Customer Category:**

     * *Discount Based*: Offer discounts based on loyalty.
     * *Purchase History Based*: Recommend products based on browsing history.

4. **Send Emails:**

   * Click **Generate and Send Emails**.
   * AI will generate personalized email content and images.
   * Emails are sent to selected recipients.
   * Progress messages show number of emails sent and failures.

---

## License 

This project is licensed under the MIT License - see the LICENSE file for details.
