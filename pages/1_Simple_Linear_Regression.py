import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

from scipy.stats import multivariate_normal

st.set_page_config(layout="wide")

if 'data' not in st.session_state:
    # Generate some data
    # The true model is y = -1/1000*x^3 + 1/10*x^2 - 5*x + 10
    x_obs = np.random.normal(50, 20, 100)
    y_obs = -1/1000*x_obs**3 + 1/10*x_obs**2 - 5*x_obs + 10 + np.random.normal(0, 20, 100)
    
    st.session_state['data'] = pd.DataFrame({'x': x_obs, 'y': y_obs})

# Let's create a linear regression class
from scipy.stats import t, f

class LinearRegression:
    def __init__(self, intercept=True) -> None:
        self.X = None
        self.y = None
        self.beta = None
        self.residuals = None
        self.sigma2 = None
        self.beta_vars = None
        self.beta_tscores = None
        self.beta_probability = None
        self.intercept = intercept
        
    def fit(self, X, y):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        self.X = X
        self.y = y
        design_matrix = X.T @ X
        inverse_design = np.linalg.inv(design_matrix)
        self.design_matrix = design_matrix
        self.inverse_design = inverse_design
        self.beta = inverse_design @ X.T @ y
        self.residuals = y - X @ self.beta
        self.sigma2 = np.sum(self.residuals**2) / (len(y) - X.shape[1] - 1)
        self.beta_vars = inverse_design * self.sigma2
        self.beta_tscores = self.beta / np.sqrt(np.diag(self.beta_vars).flatten())
        self.beta_probability = 2 * (1 - t.cdf(np.abs(self.beta_tscores), len(y) - X.shape[1] - 1))
        
    def predict(self, X, confidence_interval=False, prediction_interval=False):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        y_pred = X @ self.beta
        if confidence_interval and prediction_interval:
            dof = len(self.y) - X.shape[1] - 1
            s = np.sqrt(self.sigma2)
            t_crit = t.ppf(confidence_interval + (1 - confidence_interval) / 2, dof)
            ci = t_crit * s * np.sqrt(X @ self.inverse_design @ X.T)
            pi = t_crit * s * np.sqrt(1 + X @ self.inverse_design @ X.T)
            return y_pred, np.diag(ci).flatten(), np.diag(pi).flatten()
        elif confidence_interval:
            dof = len(self.y) - X.shape[1]
            s = np.sqrt(self.sigma2)
            t_crit = t.ppf(confidence_interval + (1 - confidence_interval) / 2, dof)
            x_mean = np.mean(self.X[:,1:], axis=0)
            Sxx = np.sum((self.X[:,1:] - x_mean)**2, axis=0)
            ci = t_crit * s * np.sqrt(1/len(self.y) + (X[:,1:] - x_mean)**2 / Sxx)
            return y_pred, ci
        elif prediction_interval:
            dof = len(self.y) - X.shape[1]
            s = np.sqrt(self.sigma2)
            t_crit = t.ppf(confidence_interval + (1 - confidence_interval) / 2, dof)
            x_mean = np.mean(self.X[:,1:], axis=0)
            Sxx = np.sum((self.X[:,1:] - x_mean)**2, axis=0)
            pi = t_crit * s * np.sqrt(1 + 1/len(self.y) + (X[:,1:] - x_mean)**2 / Sxx)
            return y_pred, pi
        return y_pred
    
    def summary(self):
        print('Linear Regression Summary')
        print('==========================')
        print(f'Number of observations: {len(self.y)}')
        print(f'Number of predictors: {self.X.shape[1]}')
        print(f'Sigma^2: {self.sigma2:.4f}')
        r2 = 1 - np.sum(self.residuals**2) / np.sum((self.y - np.mean(self.y))**2)
        print(f'R^2: {r2:.4f}')
        print(f'Adjusted R^2: {1 - (1 - r2) * (len(self.y) - 1) / (len(self.y) - self.X.shape[1] - 1):.4f}')
        print(f'F-statistic: {(np.sum(self.residuals**2) / (self.X.shape[1] - 1)) / self.sigma2:.4f}')
        print(f'p-value: {1 - f.cdf((np.sum(self.residuals**2) / (self.X.shape[1] - 1)) / self.sigma2, self.X.shape[1] - 1, len(self.y) - self.X.shape[1] - 1):.4f}')
        print("====================================")
        print('Coefficients')
        for i, beta in enumerate(self.beta):
            print(f'beta_{i}: {beta:.4f}')
            print(f't-score: {self.beta_tscores[i]:.4f}')
            print(f'p-value: {self.beta_probability[i]:.4f}')
            print()
            
    def summary_streamlit(self):
        r2 = 1 - np.sum(self.residuals**2) / np.sum((self.y - np.mean(self.y))**2)
        txt = f"""
        ### Linear Regression Summary

        Number of observations: {len(self.y)}  
        Number of predictors: {self.X.shape[1]}  
        $\sigma^2$: {self.sigma2:.4f}  
        $R^2$: {r2:.4f}  
        Adjusted $R^2$: {1 - (1 - r2) * (len(self.y) - 1) / (len(self.y) - self.X.shape[1] - 1):.4f}  
        F-statistic: {(np.sum(self.residuals**2) / (self.X.shape[1] - 1)) / self.sigma2:.4f}  
        p-value: {1 - f.cdf((np.sum(self.residuals**2) / (self.X.shape[1] - 1)) / self.sigma2, self.X.shape[1] - 1, len(self.y) - self.X.shape[1] - 1):.4f}  
        Coefficients:  """
        for i, beta in enumerate(self.beta):
            txt += fr"""
            ----------------------  
            $\beta_{i}$: {beta:.4f}  
            t-score: {self.beta_tscores[i]:.4f}  
            p-value: {self.beta_probability[i]:.4f}  """
        return txt
            
            
def main():
    st.title('Simple Linear Regression')
    
    st.markdown(
        """
        Data is generated from the following model:   
        $$y = -\\frac{1}{1000}x^3 + \\frac{1}{10}x^2 - 5x + 10 + \\epsilon$$,
        where $$\\epsilon \\sim N(0, 20^2)$$.    
        """)
    
    # Draw the generated data
    x_obs = st.session_state['data']['x']
    y_obs = st.session_state['data']['y']

    bases = st.selectbox('Select basis functions', ['Polynomial', 'Gaussian', 'Sigmoid', 'Sinusoidal'])
    
    if bases == 'Polynomial':
        st.markdown(
            """
            Play around with the polynomial degree to see how the model changes.
            - Since we know the true model is a cubic polynomial, we expect the model to perform best when the degree is 3.   
            - Notice how the credibility interval increases as the density of the observations decreases.
            - Cases overfitting and underfitting can be seen when the degree is too high or too low.
            - Check out the model's summary statistics to see how the model performs.
            """
        )
        
        degree = st.slider('Polynomial degree', 1, 10, 1)
        
        X = np.zeros((len(x_obs), degree))
        for i in range(degree):
            X[:,i] = x_obs**(i+1)

        model = LinearRegression()
        model.fit(X, y_obs)
        
        xs = np.linspace(np.min(x_obs), np.max(x_obs), 100)
        xs_design = np.zeros((len(xs), degree))
        for i in range(degree):
            xs_design[:,i] = xs**(i+1)
    elif bases == 'Gaussian':
        st.markdown(
            """
            A Gaussian basis function is defined as: $$\\phi_j(x) = \\exp\\left(-\\frac{(x - \\mu_j)^2}{2\\sigma^2}\\right)$$.
            - The centres of the basis functions are evenly spaced from 0 to 100.
            - The variance of the basis functions are all the same.
            - Play around with the number of centres and the variance to see how the model changes.
            - While the model is able to fit the data well, it is not able to capture the true model and extrapolate well.
            - Increased variance acts as a regularizer, allowing each basis function to capture more of the data.
            """
        )
        
        num_centres = st.slider('Number of centres', 1, 20, 1)
        variance = st.slider('Variance', 1, 100, 1)
        
        centres = np.linspace(0, 100, num_centres)
        variances = np.ones(num_centres) * variance
        X = np.zeros((len(x_obs), num_centres))
        for i in range(num_centres):
            X[:,i] = np.exp(-(x_obs - centres[i])**2 / variances[i] / 2)
        
        model = LinearRegression()
        model.fit(X, y_obs)
        
        xs = np.linspace(np.min(x_obs), np.max(x_obs), 100)
        xs_design = np.zeros((len(xs), num_centres))
        for i in range(num_centres):
            xs_design[:,i] = np.exp(-(xs - centres[i])**2 / variances[i] / 2)
    elif bases == 'Sigmoid':
        st.markdown(
            """
            A sigmoid basis function is defined as: $$\\phi_j(x) = \\frac{1}{1 + \\exp\\left(-(x - \\mu_j)\\right)}$$.
            - The centres of the basis functions are evenly spaced from 0 to 100.
            - Interesting to see is the step-like behaviour of the basis functions.
            - Sigmoid basis functions perform much worse than Gaussian basis functions.
            - Relatively easy to overfit the data.
            - Usually bad at extrapolating.
            """
        )
        
        num_sigmoids = st.slider('Number of sigmoids', 1, 30, 1)
        X = np.zeros((len(x_obs), num_sigmoids))
        for i in range(num_sigmoids):
            X[:,i] = 1 / (1 + np.exp(-(x_obs - i*100/num_sigmoids)))
        
        model = LinearRegression()
        model.fit(X, y_obs)
        
        xs = np.linspace(np.min(x_obs), np.max(x_obs), 100)
        xs_design = np.zeros((len(xs), num_sigmoids))
        for i in range(num_sigmoids):
            xs_design[:,i] = 1 / (1 + np.exp(-(xs - i*100/num_sigmoids)))
    elif bases == 'Sinusoidal':
        st.markdown(
            """
            The sinusoidal basis functions are defined as: $$\\phi_j(x) = \\sin\\left(\\frac{j \\pi x}{100}\\right)$$.
            - The number of frequencies can be changed to see how the model changes.
            - Arguably very similar to a Fourier series.
            - A decent fit to the data can be achieved with a small number of frequencies.
            - Overfitting can be seen when the number of frequencies is too high.
            """
        )
        
        num_freqs = st.slider('Number of frequencies', 1, 30, 1)
        X = np.zeros((len(x_obs), num_freqs))
        for i in range(num_freqs):
            X[:,i] = np.sin(x_obs * (i+1) * np.pi / 100)
        
        model = LinearRegression()
        model.fit(X, y_obs)
        
        xs = np.linspace(np.min(x_obs), np.max(x_obs), 100)
        xs_design = np.zeros((len(xs), num_freqs))
        for i in range(num_freqs):
            xs_design[:,i] = np.sin(xs * (i+1) * np.pi / 100)
    
    ys, ci, pi = model.predict(xs_design, confidence_interval=0.95, prediction_interval=0.95)
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(x_obs, y_obs, '.', markersize=5, label='Data')
    plt.plot(xs, ys, label='Line of best fit')
    plt.fill_between(xs.flatten(), ys-ci, ys+ci, alpha=0.2, label=r'95% CI')
    plt.fill_between(xs.flatten(), ys-pi, ys+pi, alpha=0.2, label=r'95% PI')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.ylim(np.min(y_obs) - 100, np.max(y_obs) + 100)
    st.pyplot(fig)
    
    st.markdown(model.summary_streamlit())
    
main()