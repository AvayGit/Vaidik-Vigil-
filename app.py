from flask import Flask, render_template, request
import joblib
import plotly.graph_objects as go
import pandas as pd

app = Flask(__name__)

# Load the saved ARIMA model
arima_model = joblib.load('model.pkl')

# Load historical crime data for the states
crime_data = pd.read_csv('crime.csv')  # Make sure you have this file with historical data

@app.route('/')
def index():
    states = crime_data['STATE/UT'].unique()  # Get the list of states from the dataset
    return render_template('index.html', states=states)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        years_to_predict = int(request.form.get('years'))
        selected_state = request.form.get('state')

        # Validate that the number of years is a positive integer
        if years_to_predict <= 0:
            raise ValueError("The number of years must be positive.")

        # Get historical data for the selected state
        state_data = crime_data[crime_data['STATE/UT'] == selected_state]

        # Forecast crime rates based on historical data for the state
        forecast = arima_model.forecast(steps=years_to_predict, exog=state_data)  # Use exogenous vars as needed
        forecast_values = [round(val, 2) for val in forecast.tolist()]  # Round to 2 decimal places

        # Create a Plotly graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, years_to_predict + 1)),
            y=forecast_values,
            mode='lines+markers',
            name='Predicted Crime Rates'
        ))

        # Add title and labels
        fig.update_layout(
            title=f'Predicted Crime Rates in {selected_state} (Next {years_to_predict} Years)',
            xaxis_title='Year',
            yaxis_title='Crime Rates (units per 100,000 population)',  # Change units as needed
            yaxis=dict(tickformat=".2f")  # Format y-axis with two decimals
        )

        # Convert Plotly figure to HTML
        graph_html = fig.to_html(full_html=False)

        # Return the forecasted values and the Plotly graph
        return render_template('index.html', prediction=forecast_values, years=years_to_predict, graph_html=graph_html, selected_state=selected_state)

    except ValueError as e:
        # If there's an error (e.g., invalid input or negative years), display the error message
        return render_template('index.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
