import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Model
import itertools
import os
import seaborn as sns
from scipy.integrate import simps
import math

font = {'family': 'serif',
        'weight': 'bold',
        'size': 18}

matplotlib.rc('font', **font)

plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('legend', fontsize='x-small')    # legend fontsize


class GrowthFit():
    def __init__(self, file_name, name='', output_loc='output/'):
        self.file_name = file_name
        if name is '':
            self.name = self.file_name[:-5]
        else:
            self.name = name

        self.output_loc = output_loc

        # Intitialize data read and get xs and ys
        data = pd.read_excel(self.file_name)
        headers = list(data)
        self.x = data[headers[0]]  # time, concentration, etc
        self.ys = [data[headers[i]]
                   for i in range(1, len(headers), 2)]  # data points
        self.errors = [data[headers[i]] for i in range(2, len(headers), 2)]
        # Create models
        self.models = [Model(self.__logistic), Model(self.__gompertz), Model(
            self.__mod_gompertz), Model(self.__richards)]

        self.results = []
        self.delys = []
        self.areas = []
        self.result_dataframe = None

    def __logistic(self, t, a, mu, lamb):
        return a/(1+np.exp(4*mu/a*(lamb-t)+2))

    def __gompertz(self, t, a, mu, lamb):
        return a*np.exp(-1*np.exp(mu*np.e/a*(lamb-t)+1))

    def __mod_gompertz(self, t, a, mu, lamb, alph, t_shift):
        return self.__gompertz(t, a, mu, lamb) + a*np.exp(alph*(t-t_shift))

    def __richards(self, t, a, mu, lamb, nu):
        return a*(1+nu*np.exp(1+nu+mu/a*(1+nu)**(1+1/nu)*(lamb-t)))**(-1/nu)

    def fit(self, sigma=2):
        self.sigma = sigma
        # Go over all data sets
        for y in self.ys:
            result = None
            dely = None
            aic = 1E6
            area = 0
            # For each of the models, fit the functions to calculate AIC
            # for model in self.models[1]:
            model = self.models[0]
            params = {}
            for name in model.param_names:
                params[name] = 1
            try:
                model_result = model.fit(y, t=self.x, **params)
                # AIC = Akaike Information Criterion,https://en.wikipedia.org/wiki/Akaike_information_criterion
                if(model_result.aic < aic):  # pylint: disable=no-member
                    result = model_result
                    aic = model_result.aic  # pylint: disable=no-member
                    dely = model_result.eval_uncertainty(sigma=self.sigma)
                    area = simps(result.best_fit, x=self.x)
            except (ValueError, TypeError):
                pass
            self.results.append(result)
            self.delys.append(dely)
            self.areas.append(area)

    # Creates a dataframe necessary to export excel
    def __create_df(self):
        all_models_param_results = {}
        # Go over all data
        for y, result, area in zip(self.ys, self.results, self.areas):
            errors = []
            param_results = {}
            error_results = {}

            # Calculate errors from covariance
            for i in range(len(result.init_values.keys())):
                errors.append(np.sqrt(result.covar[i, i]))

            # With those errors, make a dictionary for pandas
            for key, error in zip(result.best_values.keys(), errors):
                val = result.best_values[key]
                param_results['model'] = result.model.name[8:-1]
                param_results[key] = val
                param_results['area'] = area
                error_results[key] = error
                if(key == 'mu'):
                    param_results['doubling time'] = np.log(
                        2)/(4*val/param_results['a'])
                all_models_param_results[y.name] = param_results
                all_models_param_results[f'{y.name} - error'] = error_results
        self.result_dataframe = pd.DataFrame(all_models_param_results)

    def __calculate(self, param_results, best_values):
        pass

    def export(self, name=''):
        self.__create_df()
        # Check to see if fit has been done yet.
        # Result will be populated
        if(len(self.results) > 0):
            self.__make_dir()
            if name is '':
                name = f'{self.output_loc}{self.name}-output.xlsx'
            elif name.find('.xlsx') == -1:
                name = f'{self.output_loc}{name}.xlsx'
            else:
                name = f'{self.output_loc}{name}'

            self.result_dataframe.to_excel(name)

    def plot(self, save_fig=True, file_name='', ext='png', show=True):
        # Check to see if fit has been done yet.
        # Result will be populated
        num = len(self.results)
        if(num > 0):

            # Make one figure to plot
            fig = plt.figure(figsize=(8, 6))
            # Marker for plots, iterates through these marks
            marker = itertools.cycle(('o', 'v', '^', '<', '>', 's', '8', 'p'))
            lines = itertools.cycle(('-', '--', '-.', ':'))
            for y, result, dely, error in zip(self.ys, self.results, self.delys, self.errors):

                # Best fit
                vals = result.best_values
                name = result.model.name[8:-1]
                func = None

                if(name == 'logistic'):
                    func = self.__logistic
                elif(name == 'gompertz'):
                    func = self.__gompertz
                elif(name == 'mod_gompertz'):
                    func = self.__mod_gompertz
                elif (name == 'richards'):
                    func = self.__richards

                x = np.linspace(0, max(self.x)*1.02, 1000)
                y_val = func(x, **vals)

                # Plot actual data
                plt.errorbar(self.x, y, yerr=error,
                             c='k', marker=next(marker), linestyle='None')

                plt.plot(
                    x, y_val, c='k', label=f'{y.name} fit', linestyle=next(lines))

            plt.ylim([0.8, max([max(y) for y in self.ys])*1.1])
            plt.xlim([0, max(self.x)*1.02])
            plt.legend()
            plt.xlabel('Days')
            plt.ylabel('Relative Growth')
            plt.tight_layout()

            if(save_fig):
                self.__make_dir()
                if file_name is '':
                    file_name = f'{self.output_loc}{self.name}-output.{ext}'
                elif file_name.find(ext) == -1:
                    file_name = f'{self.output_loc}{file_name}.{ext}'
                else:
                    file_name = f'{self.output_loc}{file_name}'
                fig.savefig(file_name)
            if(show):
                plt.show()
        else:
            print('Need to fit first!')

    def plot_together(self, nums, save_fig=True, file_name='', ext='png', show=True):

        num = len(self.results)
        if(num > 0):
            # Make one figure to plot
            fig = plt.figure()
            # Marker for plots, iterates through these marks
            marker = itertools.cycle(('o', 'v', '^', '<', '>', 's', '8', 'p'))
            lines = itertools.cycle(('-', '--', '-.', ':'))
            # Used for choosing color

            ys = []
            results = []
            delys = []
            errors = []
            for num in nums:
                ys.append(self.ys[num])
                results.append(self.results[num])
                delys.append(self.delys[num])
                errors.append(self.errors[num])

            for y, result, dely, error in zip(ys, results, delys, errors):
                # Best fit
                vals = result.best_values
                name = result.model.name[8:-1]
                func = None

                if(name == 'logistic'):
                    func = self.__logistic
                elif(name == 'gompertz'):
                    func = self.__gompertz
                elif(name == 'mod_gompertz'):
                    func = self.__mod_gompertz
                elif (name == 'richards'):
                    func = self.__richards

                x = np.linspace(0, max(self.x)*1.02, 1000)
                y_val = func(x, **vals)

                # Plot actual data
                plt.errorbar(self.x, y, yerr=error,
                             c='k', marker=next(marker), linestyle='None')

                plt.plot(
                    x, y_val, c='k', label=f'{y.name} fit', linestyle=next(lines))

            plt.ylim([0, max([max(y) for y in self.ys])*1.1])
            plt.xlim([0, max(self.x)*1.02])
            plt.legend()
            plt.xlabel('Days')
            plt.tight_layout()

            if(save_fig):
                self.__make_dir()
                if file_name is '':
                    file_name = f'{self.output_loc}{self.name}-output.{ext}'
                elif file_name.find(ext) == -1:
                    file_name = f'{self.output_loc}{file_name}.{ext}'
                else:
                    file_name = f'{self.output_loc}{file_name}'
                fig.savefig(file_name)
            if(show):
                plt.show()

        pass

    def __make_dir(self):
        if not os.path.exists(self.output_loc):
            os.mkdir(self.output_loc)

    def get_survival_fraction(self, delay_time=0, max_slope=False, save=True, filename=''):
        survival_fraction = {}

        # Doubling time of best fit:
        result = self.results[0]
        y = self.ys[0]
        func = None

        vals = result.best_values
        name = result.model.name[8:-1]
        func = None

        if(name == 'logistic'):
            func = self.__logistic
        elif(name == 'gompertz'):
            func = self.__gompertz
        elif(name == 'mod_gompertz'):
            func = self.__mod_gompertz
        elif (name == 'richards'):
            func = self.__richards

        x = np.linspace(0, max(self.x), 1000000)
        y_val_init = func(x, **vals)
        doubling_time = np.log(2)/(4*vals['mu']/vals['a'])
        double_error = np.sqrt((np.sqrt(
            result.covar[1, 1])/vals['mu'])**2 + (np.sqrt(result.covar[0, 0])/vals['a'])**2)

        # Find max slope (Supposedly in the exponential state and calculate SF from that.)
        if max_slope:
            i_max = np.argmax(np.abs(np.gradient(y_val_init)))
            init_val = y_val_init[i_max]
            delay_time = x[i_max]
        # Or use user-defined values.
        else:
            if delay_time == 0:
                delay_time = doubling_time
            index = int(round(delay_time/np.diff(x)[0]))
            init_val = y_val_init[index]

        print(init_val)
        for result, y in zip(self.results, self.ys):
            vals = result.best_values
            y_val = func(x, **vals)
            time = self.___find_nearest(y_val, init_val)*np.diff(x)[0]
            SF = np.exp(-(time-delay_time)/doubling_time)
            survival_fraction[y.name] = {
                'SF': SF, 'error': double_error*SF/np.sqrt(len(y)), 'dt': doubling_time}

        sf_df = pd.DataFrame.from_dict(survival_fraction)

        self.__make_dir()
        if save:
            if filename is '':
                filename = f'{self.output_loc}{self.name}-sf-output.xlsx'
            elif filename.find('.xlsx') == -1:
                filename = f'{self.output_loc}{filename}.xlsx'
            else:
                filename = f'{self.output_loc}{filename}'

            sf_df.to_excel(filename)

        return survival_fraction

    def ___find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx
