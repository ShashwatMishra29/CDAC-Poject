# IMPORTING LIBRARIES
# export PATH="$PATH:/c/Users/vansh/AppData/Roaming/Python/Python312/Scripts"

import streamlit as st
from streamlit_option_menu import option_menu
# Standard library imports
import os
import pickle

# Third-party library imports
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Statistical data visualization

# Scikit-learn imports for preprocessing and model evaluation
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler  # Data scaling
from sklearn.model_selection import train_test_split  # Splitting data into train and test sets
from sklearn.metrics import mean_squared_error  # Model evaluation metric
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox
# Statsmodels imports for time series analysis
from statsmodels.tsa.stattools import adfuller, kpss  # Statistical tests for stationarity
from statsmodels.tsa.arima.model import ARIMA  # ARIMA model
from statsmodels.tsa.statespace.sarimax import SARIMAX  # SARIMAX model
import statsmodels.api as sm  # General statsmodels API

# Pmdarima imports for automated ARIMA modeling
from pmdarima import auto_arima
import pmdarima as pm

# TBATS model for time series forecasting
from tbats import TBATS
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import warnings
warnings.filterwarnings("ignore")


# Load the data

data = pd.read_csv('../../Dataset/expanded_dataset.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'],  errors='coerce')
data.set_index('Datetime', inplace=True)
data['TotalPowerConsumption']= data['PowerConsumption_Zone1'] + data['PowerConsumption_Zone2'] + data['PowerConsumption_Zone3']
data= data.drop(['PowerConsumption_Zone1','PowerConsumption_Zone2','PowerConsumption_Zone3'],axis=1)
def apply_transformations(data):
    
    data['Temperature'], lambda_boxcox = boxcox(data['Temperature'])
    pt = PowerTransformer(method='yeo-johnson')
    data['Humidity'] = pt.fit_transform(data[['Humidity']])
    data['WindSpeed'], lambda_boxcox = boxcox(data['WindSpeed'])
    data['GeneralDiffuseFlows'], lambda_boxcox = boxcox(data['GeneralDiffuseFlows'])
    data['DiffuseFlows'], lambda_boxcox = boxcox(data['DiffuseFlows'])
    data['TotalPowerConsumption'], lambda_boxcox = boxcox(data['TotalPowerConsumption'])
    return data

# Scaler = StandardScaler()
# apply_transformations(data)

#x = data.drop(['TotalPowerConsumption'], axis=1)

# Set page configuration
st.set_page_config(page_title='Forecasting Electric Power Consumption', page_icon=':zap:', layout='wide', initial_sidebar_state='expanded')

# Set title
st.markdown(
    """
    <style>
    .fixed-title {
        position: fixed;
        top: 0;
        width: 100%;
        background-color: #0e1117;
        padding: 35px 0; /* Increased padding */
        z-index: 1000;
        text-align: left;
        color: white;
    }
    .content {
        padding-top: 80px; /* Adjust this value based on the height of your fixed title */
    }
    .sidebar .sidebar-content {
        width: 200px; /* Adjust the width of the sidebar */
    }
    </style>
    <div class="fixed-title">
        <h1>⚡Forecasting Electric Power Consumption</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Add some space below the fixed title
st.markdown("<br><br><br><br>", unsafe_allow_html=True)

x = data.drop(['TotalPowerConsumption'], axis=1)
y = data['TotalPowerConsumption']

with st.sidebar:
    page = option_menu(menu_title = "Main Menu", 
                        options = ['HOME','Box Plots' ,'Total Power Consumption Outlier Detection', 
                        'Data Distribution' , 'Time Series Analysis',
                        'Ad Fuller Report','ARIMA MODEL','SARIMA MODEL',
                        'EVALUATE REGRESSION MODEL','TBATS MODEL','BI DASHBOARD'],
                        icons = ['house-gear-fill','boxes','power','bar-chart-line',
                        'clock','file-earmark-bar-graph',
                        'file-earmark-bar-graph','file-earmark-bar-graph',
                        'file-earmark-bar-graph','file-earmark-bar-graph','file-earmark-bar-graph'],
                        menu_icon = "amd",
                        )


from plotly.subplots import make_subplots
import plotly.graph_objects as go

####################################################################################################
if page == 'HOME':
    st.header("Save Energy, Save Money, Save the Planet")
    st.video('./Electric.mp4')

####################################################################################################

elif page == 'Box Plots':
    
    def plot_numeric_boxplots(df, cols_per_row=3):
        """
        Create box plots for numeric columns in a DataFrame, arranged in multiple rows.
        
        Parameters:
            df (DataFrame): The input DataFrame.
            cols_per_row (int): Number of plots per row.
        
        Returns:
            fig: Plotly figure object
        """
        numeric_cols = df.select_dtypes(include='number')
        num_cols = len(numeric_cols.columns)
        rows = (num_cols // cols_per_row) + (num_cols % cols_per_row > 0)
        
        fig = make_subplots(
            rows=rows, cols=cols_per_row,
            subplot_titles=numeric_cols.columns
        )
        
        for i, col in enumerate(numeric_cols.columns):
            row = (i // cols_per_row) + 1
            col_position = (i % cols_per_row) + 1
            fig.add_trace(
                go.Box(y=numeric_cols[col], name=col, boxmean=True),
                row=row, col=col_position
            )
        
        fig.update_layout(
            height=rows * 400,
            width=cols_per_row * 200,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            showlegend=False,
            template="plotly_dark"
        )
        
        return fig

    st.header("📊 Box Plots For Entire Data")
    st.plotly_chart(plot_numeric_boxplots(data, cols_per_row=3), use_container_width=True)

    ####################################################################################################

    st.header("📊 Box Plots With Feature Selection")
    x = data.drop(['TotalPowerConsumption'], axis=1)
    y = data['TotalPowerConsumption']
    st.plotly_chart(plot_numeric_boxplots(x, cols_per_row=5), use_container_width=True)

    ####################################################################################################

    st.header("📊 Box Plots With Robust Scaling")

    RS = RobustScaler()
    scaled_x = RS.fit_transform(x)
    x = pd.DataFrame(scaled_x, columns=x.columns)
    st.plotly_chart(plot_numeric_boxplots(x, cols_per_row=5), use_container_width=True)

    ####################################################################################################

    st.header("📊 Box Plots After Reducing Skewness")
    from sklearn.preprocessing import PowerTransformer
    from scipy.stats import boxcox

    def apply_transformations(data):
        """
        Function to apply Box-Cox transformation on specified columns and Yeo-Johnson transformation on Humidity.
        
        Parameters:
        - data: DataFrame containing the data
        
        Returns:
        - data: DataFrame with transformed columns
        """
        # Apply Box-Cox transformation
        data['Temperature'], _ = boxcox(data['Temperature'])
        data['WindSpeed'], _ = boxcox(data['WindSpeed'])
        data['GeneralDiffuseFlows'], _ = boxcox(data['GeneralDiffuseFlows'])
        data['DiffuseFlows'], _ = boxcox(data['DiffuseFlows'])
        data['TotalPowerConsumption'], _ = boxcox(data['TotalPowerConsumption'])

        # Apply Yeo-Johnson transformation
        pt = PowerTransformer(method='yeo-johnson')
        data['Humidity'] = pt.fit_transform(data[['Humidity']])
        
        return data

    # Example usage
    data = apply_transformations(data)
    st.plotly_chart(plot_numeric_boxplots(data, cols_per_row=3), use_container_width=True)

####################################################################################################

elif page == 'Total Power Consumption Outlier Detection':
    st.header("📊Outlier Detection in Total Power Consumption")
    st.subheader("Without Lower and Upper Bound")

    fig = px.box(data, x='TotalPowerConsumption', title='Outlier Detection in TotalPowerConsumption')
    # Update layout for black background and box plot color
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        title=dict(font=dict(color='white'))
    )

    fig.update_traces(marker_color='red')

    st.plotly_chart(fig)
####################################################################################################

    st.header("📊Box Plot for Outlier Detection in Total Power Consumption")
    st.subheader("With Lower and Upper Bound")
    Q1 = data['TotalPowerConsumption'].quantile(0.25)
    Q3 = data['TotalPowerConsumption'].quantile(0.75)

    IQR = Q3 - Q1

    lowerbound = Q1 - (1.5 * IQR)
    upperbound = Q3 + (1.5 * IQR)

    data = data[(data.TotalPowerConsumption >= lowerbound) & (data.TotalPowerConsumption <= upperbound)]

    fig = px.box(data, x='TotalPowerConsumption', title='Outlier Detection in TotalPowerConsumption')

    # Update layout for black background and box plot color
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        title=dict(font=dict(color='white'))
    )

    # Update box plot color
    fig.update_traces(marker_color='red')

    st.plotly_chart(fig)

####################################################################################################

elif page == 'Data Distribution':
    from sklearn.preprocessing import PowerTransformer
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st
    from scipy.stats import boxcox

    def plot_distribution(data, columns, title, transformations=None):
        """
        Function to plot distribution plots for given columns with optional transformations.
        
        Parameters:
        - data: DataFrame containing the data
        - columns: List of column names to plot
        - title: Title for the Streamlit header
        - transformations: Dictionary of column names and their corresponding transformation functions
        """
        if transformations:
            for col, transform in transformations.items():
                if transform == 'boxcox':
                    data[col], _ = boxcox(data[col])
                elif transform == 'yeo-johnson':
                    pt = PowerTransformer(method='yeo-johnson')
                    data[col] = pt.fit_transform(data[[col]])

        st.header(title)

        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))

        for i, col in enumerate(columns):
            row, col_idx = divmod(i, 2)
            sns.histplot(data[col], kde=True, color='cyan', ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Distribution Plot of {col}', color='white')
            axes[row, col_idx].set_xlabel(col, color='white')
            axes[row, col_idx].set_ylabel('Density', color='white')
            axes[row, col_idx].set_facecolor('black')
            axes[row, col_idx].title.set_color('white')
            axes[row, col_idx].xaxis.label.set_color('white')
            axes[row, col_idx].yaxis.label.set_color('white')
            axes[row, col_idx].tick_params(axis='x', colors='white')
            axes[row, col_idx].tick_params(axis='y', colors='white')
            for spine in axes[row, col_idx].spines.values():
                spine.set_edgecolor('white')

        # Set dark background for the figure
        fig.patch.set_facecolor('black')

        # Adjust layout
        plt.tight_layout()

        # Show plot
        st.pyplot(fig)

    # Columns to plot
    columns = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 'TotalPowerConsumption']

    # Plot without transformation
    plot_distribution(data, columns, "📊 Distribution Plots Without Transformation")

    # Plot with transformations
    transformations = {
        'Temperature': 'boxcox',
        'Humidity': 'yeo-johnson',
        'WindSpeed': 'boxcox',
        'GeneralDiffuseFlows': 'boxcox',
        'DiffuseFlows': 'boxcox',
        'TotalPowerConsumption': 'boxcox'
    }
    plot_distribution(data, columns, "📊 Distribution Plots With Reduced Skewness", transformations=transformations)

####################################################################################################

elif page == 'Time Series Analysis':

    y = data['TotalPowerConsumption']
    import plotly.graph_objects as go

    # Filter the data for the specified date range
    filtered_data = y['2018-12-01 00:00:00':]

    st.header('📈 Total Power Consumption Time Series')

    # Create the figure
    fig = go.Figure()

    # Add the trace
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data, mode='lines', line=dict(color='red')))

    # Update the layout for dark background
    fig.update_layout(
        title='Total Power Consumption Time Series',
        xaxis_title='Datetime',
        yaxis_title='Power Consumption',
        width=1800,
        height=700,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Show the figure in Streamlit
    st.plotly_chart(fig)

    ####################################################################################################
    
    st.header('📈 Time Series Correlogram')
    def plot_correlogram(y):
        """
        Plot the correlogram of a time series using Plotly.

        Parameters:
        y (pd.Series): The time series data.
        """
        # Calculate ACF
        fas_d = sm.tsa.acf(y, nlags=200)
        fas_s = sm.tsa.acf(y.resample('D').mean(), nlags=30)

        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=('ACF', 'ACF (diary average)'))

        # Plot ACF with markers and lines to x-axis
        for i in range(len(fas_d)):
            fig.add_trace(go.Scatter(
                x=[i, i],
                y=[0, fas_d[i]],
                mode='lines',
                line=dict(color='blue'),
                showlegend=False
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=list(range(len(fas_d))),
            y=fas_d,
            mode='markers',
            marker=dict(color='blue'),
            showlegend=False
        ), row=1, col=1)

        # Plot ACF (diary average) with markers and lines to x-axis
        for i in range(len(fas_s)):
            fig.add_trace(go.Scatter(
                x=[i, i],
                y=[0, fas_s[i]],
                mode='lines',
                line=dict(color='orange'),
                showlegend=False
            ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=list(range(len(fas_s))),
            y=fas_s,
            mode='markers',
            marker=dict(color='orange'),
            showlegend=False
        ), row=1, col=2)

        # Update layout
        fig.update_layout(
            title='Time series correlogram',
            height=600,
            width=1000,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )

        fig.update_xaxes(title_text='n_lags', row=1, col=1)
        fig.update_xaxes(title_text='n_lags / n_days', row=1, col=2)

        fig.update_yaxes(title_text='ACF', row=1, col=1)
        fig.update_yaxes(title_text='ACF', row=1, col=2)

        # Show plot in Streamlit
        st.plotly_chart(fig)

    plot_correlogram(y)

####################################################################################################

elif page == 'Ad Fuller Report':

    Scaler = StandardScaler()
    x_scaled = Scaler.fit_transform(x)
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,shuffle=False)

    def adf_test(timeseries):
        print("Results of Dickey-Fuller Test:")
        dftest = adfuller(timeseries, autolag="AIC")
        dfoutput = pd.Series(dftest[0:4], index=["Test Statistic", "p-value",
                                                "Lags Used", "Number of Observations Used"])
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value
        return dfoutput

    def kpss_test(timeseries):
        print("Results of KPSS Test:")
        kpsstest = kpss(timeseries, regression="c", nlags="auto")
        kpss_output = pd.Series(
            kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"])
        for key, value in kpsstest[3].items():
            kpss_output["Critical Value (%s)" % key] = value
        return kpss_output

    st.header('📊 ADF Test Results')
    st.table(adf_test(y_train))
    st.subheader("Differentiations needed according to ADF:")
    st.text(pm.arima.utils.ndiffs(y_train, test='adf'))

    st.header('📊 KPSS Test Results')
    st.table(kpss_test(y_train))
    st.subheader("\nDifferentiations needed according to KPSS:")
    st.text(pm.arima.utils.ndiffs(y_train, test='kpss'))

####################################################################################################

elif page == 'ARIMA MODEL':
    Data_Ar=apply_transformations(data)
    
    x = Data_Ar.drop(['TotalPowerConsumption'], axis=1)
    y = Data_Ar['TotalPowerConsumption']
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,shuffle=False)

    st.header('📊 ARIMA Model')

    path = '../Models/search_sarima_model_auto.pkl'
    if os.path.exists(path):
        st.text("Model already trained and saved at : ")
        st.text(path)
        with open(path , 'rb') as pkl_file:
            a_auto_model = pickle.load(pkl_file)
    else:
        a_auto_model = pm.auto_arima(y_train, start_p=1, start_q=1, d=1,D=1,
                                        max_p=10, max_q=10, max_d=10,
                                        max_order=None,
                                        m=7,
                                        seasonal=False,
                                        test='adf',
                                        n_jobs=-1,
                                        trace=True,
                                        error_action='ignore')

        with open(path, 'wb') as pkl_file:
            pickle.dump(a_auto_model, pkl_file)
    
    a_auto_model.fit(y_train)
    arima_pred = a_auto_model.predict(n_periods=len(y_test))
    st.subheader('ARIMA Model Predictions')

    st.markdown("Arima Summary : ")
    st.code(a_auto_model.summary(), language='python')

    def calculate_all_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        # print(mse, mae, rmse, r2)
        return mse, mae, rmse, r2

    mse, mae, rmse, r2 = calculate_all_metrics(y_test, arima_pred)
    st.subheader('Model Evaluation Metrics')
    st.write(f'Mean Squared Error (ARIMA): {mse}')
    st.write(f'Mean Absolute Error (ARIMA): {mae}')
    st.write(f'Root Mean Squared Error (ARIMA): {rmse}')
    st.write(f'R-squared (ARIMA): {r2}')

    def plot_residuals(auto_model):
        """
        Plot the residuals of the Auto ARIMA model using Plotly.

        Parameters:
        auto_model (object): The fitted Auto ARIMA model.
        """
        # Calculate residuals
        residuals = auto_model.resid()

        # Create figure
        fig = go.Figure()

        # Plot residuals
        fig.add_trace(go.Scatter(
            x=residuals.index,
            y=residuals,
            mode='lines',
            name='Residuals',
            line=dict(color='blue')
        ))

        # Update layout
        fig.update_layout(
            title='Residuals of the Auto ARIMA Model',
            xaxis_title='Index',
            yaxis_title='Residuals',
            height=600,
            width=1000,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )

        # Show the figure in Streamlit
        st.plotly_chart(fig)

        # Example usage
        # plot_residuals(auto_model)

    st.subheader('Residuals of the Auto ARIMA Model')
    plot_residuals(a_auto_model)

####################################################################################################

elif page == 'SARIMA MODEL':

    Data_Ar=apply_transformations(data)
    
    x = Data_Ar.drop(['TotalPowerConsumption'], axis=1)
    y = Data_Ar['TotalPowerConsumption']
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,shuffle=False)

    st.header('📊 SARIMA Model')
    def SARIMAX(tru):
        if os.path.exists('../Models/search_sarima_model_auto.pkl') and tru:
            st.text("Model already trained and saved at ")
            st.text('../Models/search_sarima_model_auto.pkl')
            with open('../Models/search_sarima_model_auto.pkl', 'rb') as pkl_file:
                auto_model = pickle.load(pkl_file)
        else:
            auto_model = pm.auto_arima(y_train,d=1, seasonal=True, trace=True, error_action='ignore',suppress_warnings=True)
            with open('../Models/search_sarima_model_auto.pkl', 'wb') as pkl_file:
                pickle.dump(auto_model, pkl_file)
        return auto_model

    auto_model = SARIMAX(True)

    auto_model.fit(y_train)

    st.markdown("SARIMA Summary : ")
    st.code(auto_model.summary(), language='python')

    sarimax_pred = auto_model.predict(n_periods=len(y_test))

    def calculate_all_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return mse, mae, rmse, r2

    mse, mae, rmse, r2 = calculate_all_metrics(y_test, sarimax_pred)
    st.subheader('Model Evaluation Metrics')
    st.write(f'Mean Squared Error (SARIMAX): {mse}')
    st.write(f'Mean Absolute Error (SARIMAX): {mae}')
    st.write(f'Root Mean Squared Error (SARIMAX): {rmse}')
    st.write(f'R-squared (SARIMAX): {r2}')

    distinct_counts = len(np.unique(sarimax_pred))

    # st.subheader('Distinct Predictions')
    # st.write(f'Total distinct predictions: {distinct_counts}')

####################################################################################################

elif page == 'EVALUATE REGRESSION MODEL':
    Data_Ar=apply_transformations(data)
    
    x = Data_Ar.drop(['TotalPowerConsumption'], axis=1)
    y = Data_Ar['TotalPowerConsumption']
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,shuffle=False)
    st.header('📊 Evaluate Regression Models')

    # CALLING ARMIA MODEL
    path = '../Models/search_arima_model_auto.pkl'
    if os.path.exists(path):
        #st.text("Model already trained and saved at : ")
        #st.text(path)
        with open(path , 'rb') as pkl_file:
            a_auto_model = pickle.load(pkl_file)
    else:
        a_auto_model = pm.auto_arima(y_train, start_p=1, start_q=1, d=1,D=1,
                                        max_p=10, max_q=10, max_d=10,
                                        max_order=None,
                                        m=7,
                                        seasonal=False,
                                        test='adf',
                                        n_jobs=-1,
                                        trace=True,
                                        error_action='ignore')

        with open(path, 'wb') as pkl_file:
            pickle.dump(a_auto_model, pkl_file)
    
    a_auto_model.fit(y_train)
    arima_pred = a_auto_model.predict(n_periods=len(y_test))

    # CALLING SARIMA MODEL
    def SARIMAX():
        s = 4
        p = 1
        q = 1
        d = 0
        P = 0
        Q = 1
        D = 1
        model = ARIMA(y_train, order=(p,d,q), seasonal_order=(P,D,Q,s))
        model = model.fit(y_train)
        return model
    model = SARIMAX()
    sarimax_pred = model.get_prediction(start=0, end=len(y_test)-1)
    sarimax_pred = sarimax_pred.predicted_meansarimax_pred_values = sarimax_pred.predicted_mean





    

    # FUNCTION FOR EVALUATING REGRESSION MODEL
    def evaluate_regression_models(x_test, y_test, arima_pred, sarimax_pred, step=5000):
            """
            Evaluate regression models and plot the results using Plotly.

            Parameters:
            x_test (pd.Index): Index of the test set.
            y_test (pd.Series): Actual values of the test set.
            arima_pred (pd.Series): Predicted values from the ARIMA model.
            sarimax_pred (pd.Series): Predicted values from the SARIMAX model.
            step (int): Step size for downsampling the data for plotting.
            """
            # Downsample data
            x_subset = x_test[::step]
            y_subset = y_test[::step]
            arima_subset = arima_pred[::step]
            sarimax_subset = sarimax_pred[::step]

            # Create figure
            fig = go.Figure()

            # Plot actual values
            fig.add_trace(go.Scatter(
                x=x_subset,
                y=y_subset,
                mode='lines+markers',
                name='Actual',
                line=dict(color='green'),  # Change color to green
                marker=dict(symbol='circle', size=6, opacity=0.6)
            ))

            # Plot ARIMA forecast
            fig.add_trace(go.Scatter(
                x=x_subset,
                y=arima_subset,
                mode='lines+markers',
                name='ARIMA Forecast',
                line=dict(color='blue', dash='dash'),
                marker=dict(symbol='square', size=6, opacity=0.8)
            ))

            # Plot SARIMAX forecast
            fig.add_trace(go.Scatter(
                x=x_subset,
                y=sarimax_subset,
                mode='lines+markers',
                name='SARIMAX Forecast',
                line=dict(color='orange', dash='dot'),
                marker=dict(symbol='diamond', size=6, opacity=0.8)
            ))

            # Update layout
            fig.update_layout(
                title='Forecast vs Actuals',
                xaxis_title='Time Periods',
                yaxis_title='Power Consumption',
                xaxis=dict(tickangle=30),
                legend=dict(x=0, y=1),
                height=600,
                width=1000,
                xaxis_showgrid=True,
                yaxis_showgrid=True,
                xaxis_gridcolor='rgba(0,0,0,0.1)',
                yaxis_gridcolor='rgba(0,0,0,0.1)',
                xaxis_griddash='dash',
                yaxis_griddash='dash',
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white')
            )
            # Show the figure in Streamlit
            st.plotly_chart(fig)
    st.subheader('Forecast vs Actuals')
    evaluate_regression_models(x_test.index, y_test, arima_pred, sarimax_pred, step=750)

    # Select the last 'num_obs' observations
    num_obs = 10  # Adjust as needed
    x_test_subset = x_test.iloc[-num_obs:]  # Keep the original index
    y_test_subset = y_test.iloc[-num_obs:]
    arima_pred_subset = arima_pred[-num_obs:]
    sarimax_pred_subset = sarimax_pred[-num_obs:]

    # Adjust x_positions
    bar_width = 0.2
    spacing = 0.1
    x_positions = np.arange(num_obs)

    # Create figure
    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Bar(
        x=x_positions,
        y=y_test_subset,
        name="Actual",
        marker_color="black",
        opacity=0.6
    ))

    # ARIMA forecast
    fig.add_trace(go.Bar(
        x=x_positions + (bar_width + spacing),
        y=arima_pred_subset,
        name="ARIMA Forecast",
        marker_color="blue",
        opacity=0.6
    ))

    # SARIMAX forecast
    fig.add_trace(go.Bar(
        x=x_positions + 2 * (bar_width + spacing),
        y=sarimax_pred_subset,
        name="SARIMAX Forecast",
        marker_color="orange",
        opacity=0.6
    ))

    # Layout adjustments
    fig.update_layout(
        title="Forecast vs Actuals",
        xaxis=dict(
            tickmode="array",
            tickvals=x_positions + (bar_width + spacing),
            ticktext=x_test_subset.index.date,
        ),
        yaxis_title="Power Consumption",
        barmode="group",
        bargap=spacing,
        template="plotly_white",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Show plot in Streamlit
    st.subheader("Forecast vs Actuals")
    st.plotly_chart(fig)
    
####################################################################################################

elif page == 'TBATS MODEL':
    
    
    st.header('📊 TBATS Model')
    Data_Ar=apply_transformations(data)
    x = Data_Ar.drop(['TotalPowerConsumption'], axis=1)
    y = Data_Ar['TotalPowerConsumption']
    train = y[:'2018-09-30 23:50:00']
    test = y['2018-10-01 00:00:00':]

    #MODEL
    estimator = TBATS(seasonal_periods=[144, 1008], use_trend=False)

    if os.path.exists('../Models/search_tvats_model_auto.pkl'):
        st.write("Model already trained and saved at ")
        st.write('../Models/search_tvats_model_auto.pkl')
        with open('../Models/search_tvats_model_auto.pkl', 'rb') as pkl_file:
            fitted_model = pickle.load(pkl_file)
    else:
        fitted_model = estimator.fit(train)
        with open('../Models/search_tvats_model_auto.pkl', 'wb') as pkl_file:
            pickle.dump(fitted_model, pkl_file)

    st.subheader('TBATS Model Summary')
    st.code(fitted_model.summary(), language='python')

    # Ensure the lengths match
    if len(fitted_model.y_hat) == len(train.index):
        pred_train = pd.Series(data=fitted_model.y_hat, index=train.index)
    else:
        pred_train = pd.Series(data=fitted_model.y_hat[:len(train.index)], index=train.index[:len(fitted_model.y_hat)])

    pred_test, conf_test = fitted_model.forecast(steps=len(test), confidence_level=0.95)
    pred_test = pd.Series(data=pred_test, index=test.index)
    conf_test = pd.DataFrame(data={'lower_bound': conf_test['lower_bound'], 'upper_bound': conf_test['upper_bound']}, index=test.index)

    # Ensure the lengths match for residuals
    if len(fitted_model.resid) == len(train.index):
        err_train = pd.Series(data=fitted_model.resid, index=train.index)
    else:
        err_train = pd.Series(data=fitted_model.resid[:len(train.index)], index=train.index[:len(fitted_model.resid)])

    err_test = test - pred_test
    err = pd.concat([err_train, err_test])

    st.subheader('TBATS Model Visualization')
    # Whole visualization
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train.index,
        y=train,
        mode='lines',
        name='Train',
        line=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=test.index,
        y=test,
        mode='lines',
        name='Test',
        line=dict(color='indianred')
    ))

    fig.add_trace(go.Scatter(
        x=pred_train.index,
        y=pred_train,
        mode='lines',
        name='Train pred',
        line=dict(color='teal')
    ))

    fig.add_trace(go.Scatter(
        x=pred_test.index,
        y=pred_test,
        mode='lines',
        name='Test pred',
        line=dict(color='turquoise')
    ))

    fig.add_trace(go.Scatter(
        x=conf_test.index,
        y=conf_test['lower_bound'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=conf_test.index,
        y=conf_test['upper_bound'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,0,0,0.1)',
        showlegend=True
    ))

    fig.update_layout(
        title='TBATS forecasting',
        xaxis_title='Datetime',
        yaxis_title='Power Consumption',
        legend=dict(x=0, y=1),
        height=600,
        width=1000,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Show plot in Streamlit
    st.plotly_chart(fig)

    st.subheader('TBATS Errors')
    # TBATS errors
    fig = make_subplots(rows=1, cols=2, subplot_titles=('TBATS errors', 'Error Distribution'))

    # Error plot
    fig.add_trace(go.Scatter(
        x=err.index,
        y=err,
        mode='markers',
        name='Errors',
        marker=dict(color='blue')
    ), row=1, col=1)

    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Power Consumption', row=1, col=1)

    # Error histogram
    fig.add_trace(go.Histogram(
        x=err,
        nbinsx=13,
        name='Error Distribution',
        marker=dict(color='blue')
    ), row=1, col=2)

    fig.update_xaxes(title_text='Power Consumption', row=1, col=2)

    fig.update_layout(
        height=600,
        width=1200,
        showlegend=False,
        title_text='TBATS errors',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Show plot in Streamlit
    st.plotly_chart(fig)

    st.subheader('Train Dataset and Forecasting Visualization')

    # Train dataset and forecasting visualization
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train['2018-8-01 00:00:00':'2018-9-15 23:50:00'].index,
        y=train['2018-8-01 00:00:00':'2018-9-15 23:50:00'],
        mode='lines',
        name='Train',
        line=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=pred_train['2018-8-01 00:00:00':'2018-9-15 23:50:00'].index,
        y=pred_train['2018-8-01 00:00:00':'2018-9-15 23:50:00'],
        mode='lines',
        name='Train pred',
        line=dict(color='teal')
    ))

    fig.update_layout(
        title='TBATS forecasting (train)',
        xaxis_title='Datetime',
        yaxis_title='Power consumption',
        legend=dict(x=0, y=1),
        height=600,
        width=1000,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Show plot in Streamlit
    st.plotly_chart(fig)

    st.subheader('TBATS Errors (Train)')
    # TBATS errors (train)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('TBATS errors (train)', 'Error Distribution'))

    # Error plot
    fig.add_trace(go.Scatter(
        x=err['2018-03-01 00:00:00':'2018-03-06 23:50:00'].index,
        y=err['2018-03-01 00:00:00':'2018-03-06 23:50:00'],
        mode='markers',
        name='Errors',
        marker=dict(color='blue')
    ), row=1, col=1)

    fig.update_xaxes(title_text='Datetime', row=1, col=1)
    fig.update_yaxes(title_text='Power Consumption', row=1, col=1)

    # Error histogram
    fig.add_trace(go.Histogram(
        x=err['2017-03-01 00:00:00':'2017-03-06 23:50:00'],
        nbinsx=13,
        name='Error Distribution',
        marker=dict(color='blue')
    ), row=1, col=2)

    fig.update_xaxes(title_text='Power Consumption', row=1, col=2)

    fig.update_layout(
        height=600,
        width=1200,
        showlegend=False,
        title_text='TBATS errors (train)',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Show plot in Streamlit
    st.plotly_chart(fig)
    
####################################################################################################

elif page == 'BI DASHBOARD':

    st.header('📊 Business Intelligence Dashboard')
    # Power BI embed URL
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=84c4e97b-18d1-41b8-b3db-25722d7788ac&autoAuth=true&ctid=6d4830a5-22e7-41cc-9c65-6c1781f1d6c7"

    # Embed Power BI dashboard using an HTML iframe
    st.components.v1.html(
        f'<iframe title="Corrected BI Dashboard" width="1500" height="600" src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>',
        height=550  # Adjust height to fit properly
    )
