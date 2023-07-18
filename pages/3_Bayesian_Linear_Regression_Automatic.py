import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

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
        

class UpdateDist:
    def __init__(self, axl, axr, a=0.5, b=0.5, noise=0.1, batch_size=5):
        self.a = a
        self.b = b
        self.noise = noise
        self.line, = axl.plot([], [], color='blue', label='Line of best fit')
        self.points, = axl.plot([], [], 'o', color='red', label='Data', markersize=5)
        self.confidence = axl.fill_between([], [], [], alpha=0.2, color='orange', label=r'$1\sigma$ PI')
        self.x = np.linspace(-10, 10, 200)
        self.axl = axl
        self.axr = axr
        self.batch_size = batch_size
        self.br = BayesianRegression(np.zeros(2), 1)

        # This line represents the theoretical value, to
        # which the plotted distribution should converge.
        self.axl.plot(self.x, a + b*self.x, 'k--', lw=1, label='True line of best fit')
        
        # Set up plot parameters
        self.axl.set_xlim(-10, 10)
        self.axl.set_ylim(-10, 10)
        self.axl.grid(True)
        # Create the legend and place it at the top left
        self.axl.legend(loc=2, prop={'size': 10})
        
        # Plot the parameter distributions
        x, y = np.mgrid[a-1:a+1:.01, b-1:b+1:.01]
        pos = np.dstack((x, y))
        rv = multivariate_normal(self.br.prior_mean, self.br.prior_cov)
        self.contour_plot = axr.contourf(x, y, rv.pdf(pos), cmap='plasma')
        self.axr.set_xlabel('$w_0$')
        self.axr.set_ylabel('$w_1$')
        self.axr.plot(self.a, self.b, 'kx', markersize=10)
        

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            self.line.set_data([], [])
            self.points.set_data([], [])
            return self.line, self.points, self.confidence

        # Create random data and update the line.
        x_obs = np.random.uniform(-10, 10, self.batch_size)
        y_obs = self.a + self.b*x_obs + np.random.normal(0, self.noise, self.batch_size)
        self.br.fit(np.vstack((np.ones(self.batch_size), x_obs)).T, y_obs)
        
        y_pred, pi = self.br.predict(np.vstack((np.ones(200), self.x)).T, pi=True)
        
        self.line.set_data(self.x, y_pred)
        self.points.set_data(x_obs, y_obs)
        self.axl.collections.clear()
        self.axl.fill_between(self.x.flatten(), y_pred-pi, y_pred+pi, alpha=0.2, color='orange', label=r'$1\sigma$ PI')
        
        # Replot the contour plot
        self.axr.collections.clear()
        x, y = np.mgrid[self.a-1:self.a+1:.01, self.b-1:self.b+1:.01]
        pos = np.dstack((x, y))
        rv = multivariate_normal(self.br.prior_mean, self.br.prior_cov)
        self.axr.contourf(x, y, rv.pdf(pos), cmap='plasma')
        
        return self.line, self.points, self.confidence
    

def main():
    
    st.title("Automatic Bayesian Linear Regression Demo")

    st.markdown(
        """
        Input the parameters of the true line that generated the data, add some noise, and watch the model converge to the true line.
        """
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        b = st.number_input('Slope', value=0.5, step=0.01)
    with col2:
        a = st.number_input('Intercept', value=0.5, step=0.01)
    with col3:
        sigma = st.number_input('Noise', value=1.0, step=0.01)
    with col4:
        batch_size = st.number_input('Batch Size', value=5, step=1)
    
    #def make_gif(a, b, sigma, batch_size):
    #    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 6))
    #    ud = UpdateDist(axl, axr, a=a, b=b, noise=sigma, batch_size=batch_size)
    #    anim = FuncAnimation(fig, ud, frames=100, interval=100, blit=True)
        
    #    components.html(anim.to_jshtml(), height=1000)   
    #st.button('Generate GIF', on_click=make_gif, args=(a, b, sigma, batch_size))
    
    if st.button('Generate GIF'):
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 6))
        ud = UpdateDist(axl, axr, a=a, b=b, noise=sigma, batch_size=batch_size)
        anim = FuncAnimation(fig, ud, frames=100, interval=100, blit=True)
        
        components.html(anim.to_jshtml(), height=1000)
    
main()