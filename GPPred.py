"""" This file contains the implementation of the ExactGP class including 
the training and only return the predictive distribution. """
import gpytorch
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample

import statsmodels.api as sm
from statsmodels.api import OLS





import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

class CalibrateModels:
    @staticmethod
    def plot_quantiles(x_test, y_test, y_test_pred, y_test_pred_std, alpha=0.05):
        """
        Plots the empirical vs. predicted quantiles to evaluate the calibration of the model.
        """
        # Calculate the CDF values of y_test based on the normal distribution defined by predictions and their standard deviation
        h_xt_yt_sm = st.norm.cdf(y_test, loc=y_test_pred, scale=y_test_pred_std)
        p_hat_sm = np.array([np.mean(h_xt_yt_sm <= p) for p in h_xt_yt_sm])
        
        # Prepare a DataFrame for easier manipulation and plotting
        results = pd.DataFrame({'x_eval': x_test, 'y_eval': y_test, 'h_xt_yt_sm': h_xt_yt_sm, 'P_hat_sm': p_hat_sm})
        display(results.head())
        
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        # set y limits to be the the min and max of the observed values
        ax.set_ylim([min(p_hat_sm), max(p_hat_sm)])
        ax.scatter(h_xt_yt_sm, p_hat_sm, alpha=0.7)
        ax.plot([0, 1], [0, 1], '--', color='grey')
        ax.set_xlabel('Predicted Quantile')
        ax.set_ylabel('Empirical Quantile')
        plt.show()

    @staticmethod
    def calibrate_model(y_test, y_test_pred, y_test_pred_std):
        """
        Calibrates the model using isotonic regression and returns the calibrator object.
        """
        h_xt_yt_sm = st.norm.cdf(y_test, loc=y_test_pred, scale=y_test_pred_std)
        p_hat_sm = np.array([np.mean(h_xt_yt_sm <= p) for p in h_xt_yt_sm])
        
        # Fit the isotonic regression model
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(h_xt_yt_sm, p_hat_sm)
        return calibrator
    
    @staticmethod
    def plot_CI(x_test, y_test, y_test_pred, lower, upper, title='Uncalibrated 95% PI', y_lim = 10):
        perc_within = np.mean((y_test <= upper)&(y_test >= lower))
        print(f'{perc_within*100:.1f}% of the points in the 95% PI')

        # plot the results
        fig, ax = plt.subplots(1, 1, figsize=(12,6))

        # set y limits to be the the min and max of the observed values
        
        ax.set_ylim([-1*y_lim, y_lim])
        ax.fill_between(x_test, y1=lower, y2=upper, color='#BDD5C8', label='PI from training set')
        ax.scatter(x_test, y_test, alpha=0.3, label='calibration set')
        ax.plot(x_test, y_test_pred, color='grey', alpha=0.7, label='model from training set')
        ax.legend(loc='upper left')
        ax.set_title(title, fontsize=17);


    def plot_calibration_curve(self, y_eval, y_eval_pred, y_eval_pred_std, y_test, y_test_pred, y_test_pred_std):
        calibrator = self.calibrate_model(y_eval, y_eval_pred, y_eval_pred_std)
        h_xt_yt_sm = st.norm.cdf(y_test, loc=y_test_pred, scale=y_test_pred_std)
        predicted_values = h_xt_yt_sm
        expected_values = np.linspace(0, 1, num=11).reshape(-1, 1)
        calibrated_values = calibrator.predict(predicted_values)

        observed_uncalibrated = np.mean(predicted_values.reshape(1, -1) <= expected_values, axis=1) 
        observed_calibrated = np.mean(calibrated_values.reshape(1, -1) <= expected_values, axis=1) 


        fig, ax = plt.subplots(nrows=1, ncols=1)
        # set y limits to be the the min and max of the observed values
        ax.set_ylim([min(observed_uncalibrated), max(observed_uncalibrated)])
        ax.plot(expected_values, observed_calibrated, 'o-', color='red', label='calibrated')
        ax.plot(expected_values, observed_uncalibrated, 'o-', color='purple', label='uncalibrated')
        ax.plot([0,1],[0,1], '--', color='black', alpha=0.7)
        ax.legend()
        
        
        # calculate the calibrated predictions
        calibrated_quantiles = calibrator.predict(h_xt_yt_sm)
        # find the inverse of the calibrated quantiles
        calibrated_y_test_pred = st.norm.ppf(calibrated_quantiles, loc=y_test_pred, scale=y_test_pred_std)
        lower_quantiles, upper_quantiles = np.clip(calibrator.predict([0.025, 0.975]), -1e-5, 1-1e-5)
        lower, upper = st.norm.ppf(lower_quantiles, loc=y_test_pred, scale=y_test_pred_std), st.norm.ppf(upper_quantiles, loc=y_test_pred, scale=y_test_pred_std)
        return calibrator, calibrated_y_test_pred, lower, upper






# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type='RBF'):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        if kernel_type == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_type == 'Matern5/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        elif kernel_type == 'Matern3/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        elif kernel_type == 'Matern1/2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        elif kernel_type == 'Linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        elif kernel_type == 'Periodic':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_type == 'Cosine':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
        elif kernel_type == 'RationalQuadratic':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQ())
        elif kernel_type == 'PiecewisePolynomialKernel':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel())
        else:
            raise ValueError('Invalid kernel type')


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




class GPPred():
    """
    This class is a wrapper around the gpytorch.models.ExactGP class. It is used to create a GP model with a
    Gaussian likelihood.
    """

    def __init__(self, train_x, train_y, kernel_type='RBF'):
            """
            This method initializes the ExactGP class.

            :param train_x: The training inputs.
            :type train_x: torch.Tensor
            :param train_y: The training targets.
            :type train_y: torch.Tensor
            :param likelihood: The likelihood function.
            :type likelihood: gpytorch.likelihoods.Likelihood
            """
            self.train_x = train_x
            self.train_y = train_y
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1.000E-06))
            # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.kernel_type = kernel_type
            self.model =  ExactGPModel(train_x, train_y, self.likelihood, self.kernel_type)
            
  
    
    def trainGP(self, n_iter):
        """
        This method trains the GP model.

        :param train_x: The training inputs.
        :type train_x: torch.Tensor
        :param train_y: The training targets.
        :type train_y: torch.Tensor
        :param likelihood: The likelihood function.
        :type likelihood: gpytorch.likelihoods.Likelihood
        :param mll: The marginal log likelihood.
        :type mll: gpytorch.mlls.ExactMarginalLogLikelihood
        :param optimizer: The optimizer.
        :type optimizer: torch.optim.Optimizer
        :param num_training_iterations: The number of training iterations.
        :type num_training_iterations: int
        """
        train_x = self.train_x
        train_y = self.train_y
        # initialize likelihood and model
       


        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(n_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            # if i % 20 == 0:
            #     print('Iter %d/%d - Loss: %.3f  ' % (
            #         i + 1, n_iter, loss.item()
                
            #     ))
            optimizer.step()
        
    def predict(self, test_x):
        """
        This method returns the predictive distribution of the GP model.

        :param test_x: The test inputs.
        :type test_x: torch.Tensor
        :param likelihood: The likelihood function.
        :type likelihood: gpytorch.likelihoods.Likelihood
        :return: The predictive distribution of the GP model.
        :rtype: gpytorch.distributions.MultivariateNormal
        """
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            observed_pred = self.model(test_x)
            observed_pred_with_noise = self.likelihood(observed_pred) 
 
        return observed_pred, observed_pred_with_noise
    




