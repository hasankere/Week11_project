from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.graph_objs as go
import plotly
import json
from forecast import generate_forecast
from portfolio import optimize_portfolio

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['GET'])
def forecast():
    forecast_data = generate_forecast()
    return jsonify(forecast_data)

@app.route('/optimize', methods=['GET'])
def optimize():
    portfolio_data = optimize_portfolio()
    return jsonify(portfolio_data)

if __name__ == '__main__':
    app.run(debug=True)
