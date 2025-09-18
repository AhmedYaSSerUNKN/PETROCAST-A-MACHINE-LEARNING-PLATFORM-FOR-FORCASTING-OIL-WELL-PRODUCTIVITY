# PETROCAST-A-MACHINE-LEARNING-PLATFORM-FOR-FORCASTING-OIL-WELL-PRODUCTIVITY

PetroCast is a Streamlit-based machine learning platform for forecasting oil well productivity. It enables users to upload production data, train and compare multiple regression models, analyze model performance, and visualize feature importance using SHAP—all through an interactive web interface.

## Features

- **Data Upload & Validation:** Upload CSV files and automatically check for required columns, missing values, and outliers.
- **Model Training:** Train models including XGBoost, Random Forest, Bayesian Ridge, SVM, and Decision Tree with time-series cross-validation.
- **Prediction:** Input feature values and predict oil production in barrels/day or liters/day.
- **Performance Metrics:** View R² and MSE for both cross-validation and test sets, with overfitting warnings.
- **Model Comparison:** Compare all supported models side-by-side.
- **Data Analysis:** Explore your data with summary statistics, box plots, and correlation heatmaps.
- **Feature Importance:** Visualize feature contributions using SHAP beeswarm plots.
- **Model Export:** Download trained models for future use.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AhmedYaSSerUNKN/PetroCast.git
   cd PetroCast
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app:**
   ```sh
   streamlit run src/Petrocast-1.py
   ```

2. **Open the app in your browser:**  
   Streamlit will provide a local URL (usually http://localhost:8501).

3. **Upload your CSV data:**  
   It depends on your file you want to upload but mine include these columns:
   - `DEPTH_MD`
   - `Reservoir_pressure`
   - `Working_hours`
   - `Oil_volume`
   - `Date`
   - `WELL`

4. **Train and evaluate models:**  
   Select a model, configure options, and click "Train Model".  
   Use the prediction interface to forecast oil production.

5. **Explore data and model results:**  
   Use checkboxes to show data analysis and compare models.

6. **Export trained models:**  
   Download your trained model as a `.pkl` file for later use.

## File Structure

```
PetroCast/
│
├── src/
│   └── Petrocast-1.py      # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies:
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - seaborn
  - shap
  - joblib

## Example Data Format

```csv
DEPTH_MD,Reservoir_pressure,Working_hours,Oil_volume,Date,WELL
2500,3200,24,120,2023-01-01,Well-1
...
```

## License

This project is licensed under the MIT License.

## Contributing

Pull requests and suggestions are welcome! Please open an issue to discuss changes.

## Contact

For questions or support, please contact me.
