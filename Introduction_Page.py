import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Linear Regression! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This web application is a collection of simple demos of 
    Linear Regression. I have implemented both the frequentist
    and Bayesian approaches to Linear Regression.
    
    ### Simple Linear Regression
    - Regress a response variable on a single predictor variable.
    - See the model's predictions, credible interval, and prediction interval.
    - Take a look at which coefficients are significant and see the model's summary statistics.
    - Change the basis functions to see how the model changes.
    - Supported basis functions: Polynomial, Gaussian, Sigmoid, Sinusoidal.
    
    ### Bayesian Linear Regression Manual Input
    - Manually place data points on the graph and see how the model changes.
    - See the posterior distribution of the model's parameters.
    
    ### Bayesian Linear Regression Automatic
    - Automatically generate data points from a given function.
    - See the posterior distribution of the model's parameters.
    
    #### The following sources were used to create this web application:
    - [A blog post by Al Popkes](https://alpopkes.com/posts/machine_learning/bayesian_linear_regression/)
    - [Another blog post by Damian Bogunowicz](https://dtransposed.github.io/2020/01/03/Bayesian-Linear-Regression/)
    - [Probability and Statistics for Engineers & Scientists; Ronald Walpole, Raymond MYers, Sharon L. Myers, Keying Ye; 9th Edition] (https://www.amazon.com/Probability-Statistics-Engineers-Scientists-Myers/dp/0321629116)
    - [Pattern Recognition and Machine Learning 2006; Christopher M. Bishop] (https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
    """
)
