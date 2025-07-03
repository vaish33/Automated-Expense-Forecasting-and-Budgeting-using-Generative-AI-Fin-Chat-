import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
import openai
from data_preparation import load_and_preprocess_data, make_stationary
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key here
openai.api_key = os.getenv('OPENAI_API_KEY')
#openai.api_key='sk-proj-DdLAwAvzYKVP392tTTIKT3BlbkFJ3ZZIHj8jZovhy7Hkx8nX'

# Load and preprocess the data
data = load_and_preprocess_data('expense_data_it_project.csv')
development_cost = make_stationary(data, 'development_cost')

def get_expense_forecast(category, months, historical_data, user_prompt):
    prompt_text = (
        f"{user_prompt}\n"
        f"Here is the historical data for {category} expenses over the last few months:\n"
        f"{historical_data}\n"
        f"Predict the {category} expenses for the next {months} months based on this data. "
        "Consider historical data, industry trends, and typical expenses related to project management activities."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=500
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"OpenAI API error: {str(e)}")
        return "Unable to generate forecast due to API quota limitations."

def generate_budgeting_recommendations(forecast):
    # Analyze the forecast data to generate recommendations
    avg_expense = forecast.mean()
    max_expense = forecast.max()
    min_expense = forecast.min()

    recommendations = f"""
    Based on the forecasted expenses:
    - **Average Monthly Expense:** ${avg_expense:.2f}
    - **Highest Expense in the Forecast Period:** ${max_expense:.2f}
    - **Lowest Expense in the Forecast Period:** ${min_expense:.2f}

    **Recommendations:**
    1. **Budget for Fluctuations:** Ensure you have a buffer in your budget to accommodate potential fluctuations.
    2. **Optimize Costs:** Explore options to optimize costs, possibly by negotiating better rates with vendors.
    3. **Monitor Regularly:** Regularly monitor actual expenses against the forecast to adjust your budget as necessary.
    4. **Prepare for Low Periods:** Utilize lower expense periods to allocate funds for future high expense periods.
    5. **Invest in Efficiency:** Consider investing in tools to automate project management activities, reducing costs.

    Implementing these recommendations can help maintain financial stability and make informed budgeting decisions.
    """
    return recommendations

# Streamlit app
st.title('Automated Expense Forecasting and Budgeting')

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

# Select expense category and forecast period
category = st.sidebar.selectbox('Select Expense Category', data.columns)
months = st.sidebar.slider('Select Forecast Period (months)', 1, 24)

# User-defined prompt
user_prompt = st.sidebar.text_area("Enter additional details or context for the forecast:", 
                                   "As a financial analyst, predict the expenses for the next few months.")

# Display historical data
st.header(f"Historical Data for {category}")
fig = px.line(data, x=data.index, y=category, title=f'{category} Over Time')
st.plotly_chart(fig)

# Generate forecast
if st.sidebar.button('Generate Forecast'):
    try:
        # Check if the data needs differencing
        if category + '_diff' in data.columns:
            model = ARIMA(data[category + '_diff'].dropna(), order=(5, 1, 0))
        else:
            model = ARIMA(data[category], order=(5, 1, 0))
        model_fit = model.fit()
        forecast_steps = months
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Plot forecast
        forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps+1, freq='M')[1:]
        forecast_series = pd.Series(forecast, index=forecast_index)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=data.index, y=data[category], mode='lines', name='Historical'))
        fig_forecast.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecast'))
        fig_forecast.update_layout(title=f'{category} Forecast', xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig_forecast)
    except Exception as e:
        st.write(f"Error fitting ARIMA model: {e}")

    try:
        historical_data = data[category].tail(12).to_string(index=False)
        forecast_text = get_expense_forecast(category, months, historical_data, user_prompt)
        st.subheader('AI-based Forecast')
        st.write(forecast_text)
    except Exception as e:
        st.write(f"Error generating forecast: {e}")

    # Generate budgeting recommendations
    recommendations = generate_budgeting_recommendations(forecast_series)
    st.subheader('Budgeting Recommendations')
    st.write(recommendations)
