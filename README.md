# Stock-Price-Trend-Prediction-with-LSTM
### Overview
This project aims to predict future stock prices using an LSTM (Long Short-Term Memory) neural network. It utilizes historical stock data to train the model and generate predictions.

### Technologies Used
* Python
* TensorFlow (Keras)
* yfinance
* Pandas
* Matplotlib
* Scikit-learn

### Prerequisites
* Python 3.x
* Required Python libraries:
    * yfinance
    * numpy
    * pandas
    * scikit-learn
    * tensorflow
    * matplotlib

### Installation

1.  Clone the repository (if applicable).
2.  Install the required Python libraries:
    ```bash
    pip install yfinance numpy pandas scikit-learn tensorflow matplotlib
    ```

### Usage

1.  **Fetch Data:** The script fetches historical stock price data from Yahoo Finance using the `yfinance` library.
2.  **Prepare Data:** The data is preprocessed by selecting the closing price, scaling it using `MinMaxScaler`, and creating sequences for the LSTM model.
3.  **Build LSTM Model:** An LSTM model is constructed using TensorFlow's Keras library.
4.  **Train and Validate Model:** The LSTM model is trained using the prepared data.
5.  **Plot Predictions vs. Actual Prices:** The script generates a plot comparing the model's predictions to the actual stock prices.
6.  **Calculate and Plot RSI:** The script calculates and plots the Relative Strength Index (RSI), a momentum indicator.
7.  **Calculate and Plot Moving Average:** The script calculates and plots the moving average of the stock prices.
8.  Run the script:
    ```bash
    python your_script_name.py
    ```
    * Modify the `symbol`, `start_date`, and `end_date` variables in the `main()` function to analyze different stocks or time periods.

### Functions

* `fetch_stock_data(symbol, start_date, end_date)`: Fetches stock price data from Yahoo Finance.
* `preprocess_data(df)`: Preprocesses the stock price data, scales it, and creates sequences for the LSTM model.
* `build_lstm_model(input_shape)`: Builds the LSTM model.
* `train_and_validate_model(model, X_train, y_train, epochs, batch_size, validation_split)`: Trains and validates the LSTM model.
* `plot_predictions(model, X_test, y_test, scaler, scaled_data, df, symbol)`: Plots the predicted stock prices against the actual prices.
* `calculate_rsi(data, window=14)`: Calculates the Relative Strength Index (RSI).
* `plot_rsi(df, symbol)`: Plots the RSI along with stock prices.
* `calculate_moving_average(data, window=20)`: Calculates the moving average.
* `plot_moving_average(df, symbol)`: Plots the moving average along with stock prices.
* `main(symbol='AAPL', start_date='2012-01-01', end_date='2024-01-01', epochs=25, batch_size=32)`: Main function to run the stock price prediction.

### Error Handling
The script includes error handling for:
* Fetching data from Yahoo Finance.
* Preprocessing the data.
* Building and training the LSTM model.
* Plotting the results.

### Improvements
* Incorporate other technical indicators.
* Improve the model architecture.
* Deploy as a web application.
