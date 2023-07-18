import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

from scipy.stats import multivariate_normal

st.set_page_config(layout="wide")

class BayesianRegression:
    def __init__(self, prior_mean, precision):
        self.prior_mean = prior_mean
        prior_cov = 1/precision * np.eye(prior_mean.shape[0])
        self.prior_cov = prior_cov
        self.X = None
        self.y = None
    
    def fit(self, X, y):
        if self.X is None:
            self.X = X
        else:
            self.X = np.vstack((self.X, X))
        if self.y is None:
            self.y = y
        else:
            self.y = np.hstack((self.y, y))
        # Do a MLE estimate of the coefficients first
        w_mle = np.linalg.solve(self.X.T.dot(self.X), self.X.T.dot(self.y))
        # Estimate beta from the residuals
        self.beta = 1/((self.y - self.X.dot(w_mle)).var())
        # Update the posterior mean and covariance
        posterior_cov_inv = np.linalg.inv(self.prior_cov) + self.beta * X.T.dot(X)
        posterior_cov = np.linalg.inv(posterior_cov_inv)
        posterior_mean = posterior_cov.dot(np.linalg.inv(self.prior_cov).dot(self.prior_mean) + self.beta * X.T.dot(y))
        self.prior_mean = posterior_mean
        self.prior_cov = posterior_cov
        
    def predict(self, X, pi=False):
        predictions = X.dot(self.prior_mean)
        if not pi:
            return predictions
        
        predictions_stds = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            predictions_stds[i] = np.sqrt(1/self.beta + x.T.dot(self.prior_cov).dot(x))
        
        return predictions, predictions_stds
    
    def plot_param_distributions(self):
        if len(self.prior_mean) > 2:
            print('Distribution plot for more than 2 parameters is not implemented yet.')
            return
        x, y = np.mgrid[-1:1:.01, -1:1:.01]
        pos = np.dstack((x, y))
        rv = multivariate_normal(self.prior_mean, self.prior_cov)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.contourf(x, y, rv.pdf(pos), cmap='plasma')
        ax.set_xlabel('$w_0$')
        ax.set_ylabel('$w_1$')
        plt.show()
        


def main():
    
    st.title("Interactive Bayesian Linear Regression Demo")
    
    
    # Empty dataframe to store data points
    # We need to add it to the website's state so that it doesn't get reset on each rerun
    if 'dots_df' not in st.session_state:
        st.session_state.dots_df = pd.DataFrame(columns=['x', 'y'])
    
    # New points that we use to update the model
    if 'new_dots_df' not in st.session_state:
        st.session_state.new_dots_df = pd.DataFrame(columns=['x', 'y'])
        
    # Empty model
    if 'model' not in st.session_state:
        st.session_state.model = BayesianRegression(np.zeros(2), 1)
        
    # Model fit flag
    if 'model_fit' not in st.session_state:
        st.session_state.model_fit = False

    st.markdown(
        """
        Manually place data points on the graph and see how the model changes.
        - Click on the "Place Dot" button to place a new data point.
        - Click on the "Remove Last Dot" button to remove the last data point.
        - Click on the "Clear Dots" button to remove all data points.
        - Click on the "Update Model" button to update the model with the new data points.
        """
    )
    
    # Sidebar controls
    x = st.slider('x', min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    y = st.slider('y', min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    
    def add_dot(x, y):
        new_dot = {'x': x, 'y': y}
        st.session_state.new_dots_df = pd.concat([st.session_state.new_dots_df, pd.DataFrame(new_dot, index=[0])], ignore_index=True)
    
    def remove_last_dot():
        st.session_state.new_dots_df = st.session_state.new_dots_df.iloc[:-1]
    
    def clear_dots():
        # Reset the old points
        st.session_state.dots_df = pd.DataFrame(columns=['x', 'y'])
        # Reset the new points
        st.session_state.new_dots_df = pd.DataFrame(columns=['x', 'y'])
        # Reset the model
        st.session_state.model = BayesianRegression(np.zeros(2), 1)
        
    def update_model():
        X = np.vstack((np.ones(st.session_state.new_dots_df.shape[0]), st.session_state.new_dots_df['x'])).T
        y = st.session_state.new_dots_df['y']
        st.session_state.model.fit(X, y)
        
        # Add the new points to the old ones
        st.session_state.dots_df = pd.concat([st.session_state.dots_df, st.session_state.new_dots_df], ignore_index=True)
        # Reset the new points
        st.session_state.new_dots_df = pd.DataFrame(columns=['x', 'y'])
        
        # Update the model fit flag
        st.session_state.model_fit = True
    
    def plot_points_predictions_param_distribution():
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].scatter(st.session_state.dots_df['x'], st.session_state.dots_df['y'], 
                   label='Fitted Points', color='blue')
        ax[0].scatter(st.session_state.new_dots_df['x'], st.session_state.new_dots_df['y'],
                   label='New Points', color='green')
        ax[0].set_xlim(-1.0, 1.0)
        ax[0].set_ylim(-1.0, 1.0)
        ax[0].set_title('Data and the predictive interval')
        # Only plot the prediction interval if there are at least 3 points
        # and the model has been updated at least once
        if len(st.session_state.dots_df) > 2 and st.session_state.model_fit:
            x = np.linspace(-1.0, 1.0, 100)
            X = np.vstack((np.ones(x.shape[0]), x)).T
            predictions, predictions_stds = st.session_state.model.predict(X, pi=True) 
            ax[0].plot(x, predictions, color='red', label='Predictions')
            ax[0].fill_between(x, predictions - predictions_stds, 
                            predictions + predictions_stds, color='red',
                            alpha=0.2, label=r'$1\sigma$ Prediction Interval')
            ax[0].legend()
        
        
        crt_mean_estimate = st.session_state.model.prior_mean
        # Draw the distribution in a 2x2 square around the current mean estimate
        x, y = np.mgrid[crt_mean_estimate[0]-1:crt_mean_estimate[0]+1:.01,
                        crt_mean_estimate[1]-1:crt_mean_estimate[1]+1:.01]
        pos = np.dstack((x, y))
        rv = multivariate_normal(st.session_state.model.prior_mean, st.session_state.model.prior_cov)
        
        ax[1].set_title('Parameter distribution')
        ax[1].contourf(x, y, rv.pdf(pos), cmap='coolwarm')
        ax[1].set_xlabel('$w_0$ - intercept')
        ax[1].set_ylabel('$w_1$ - slope')
        
        st.pyplot(fig)
        
    
    # Create three buttons side by side
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Place Dot button
        st.button('Place Dot', on_click=add_dot, args=(x, y))
    
    with col2:
        # Remove Dot button
        st.button('Remove Last Dot', on_click=remove_last_dot)
    
    with col3:
        # Clear Dots button
        st.button('Clear Dots', on_click=clear_dots)
        
    with col4:
        st.button('Update Model', on_click=update_model)
    
    plot_points_predictions_param_distribution()
    
main()