import matplotlib.pyplot as plt
import numpy as np
from control_factors import *
from math import sqrt

#vss, fix SChart
class RChart:
    def __init__(self, samples, L=3, title="R Chart"):
        self.samples = samples
        self.L = L
        self.title = title
        self.num_iterations = 1
        self.OC_samples = {}
        self.control_limits ={}
        self.__init_rChart(samples, L=L)
        self.finalLCL = self.get_final_control_limits()["LCL"]
        self.finalCL = self.get_final_control_limits()["CL"]
        self.finalUCL = self.get_final_control_limits()["UCL"]
        self.phase2_samples = []

    def show_charts(self):
        self.__rChart(self.samples, L=self.L, title=self.title)
    
    def show_phase2_chart(self):
        range_samples = np.ptp(self.phase2_samples, axis=1)
        plot(range_samples, self.finalLCL, self.finalCL, self.finalUCL, title=self.title)

    def get_num_iterations(self):
        return self.num_iterations
    
    def get_oc_samples(self):
        return self.OC_samples

    def get_control_limits(self):
        return self.control_limits
    
    def get_final_control_limits(self):
        return self.control_limits[f"Iteration {self.num_iterations}"]

    def summary(self):
        return f"Iterations: {self.num_iterations}\nControl Limits: {self.control_limits}\nOC Samples: {self.OC_samples}"
    
    def add_phase2_samples(self, samples):
        self.phase2_samples = samples
    
    def __rChart(self, samples, L = 3):
        n = len(samples[0])
        range_samples = np.ptp(samples, axis=1) 
        lcl, cl, ucl = self.__getLimits(n, range_samples, L=L) 
        plot(range_samples, lcl, cl, ucl, title=self.title)
        self.__checkWithinLimits(range_samples, lcl, cl, ucl, L=L, n=n)

    def __init_rChart(self, samples, L = 3):
        n = len(samples[0])
        range_samples = np.ptp(samples, axis=1) 
        lcl, cl, ucl = self.__getLimits(n, range_samples, L=L) 
        self.__init_checkWithinLimits(range_samples, lcl, cl, ucl, L=L, n=n)

    def __checkWithinLimits(self, samples, lcl, cl, ucl, n, L = 3):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples, L=L)
        if len(updated_samples) != len(samples):
            plot(updated_samples, lcl, cl, ucl, title=self.title)
            self.__checkWithinLimits(updated_samples, lcl, cl, ucl, n, L=L)
            
    def __init_checkWithinLimits(self, samples, lcl, cl, ucl, n, L = 3):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples, L=L)
        self.control_limits[f"Iteration {self.num_iterations}"] = {"LCL": lcl, "CL": cl, "UCL": ucl} 
        if len(updated_samples) != len(samples):
            self.OC_samples[f"Iteration {self.num_iterations}"] = OC_samples
            self.num_iterations += 1
            self.__init_checkWithinLimits(updated_samples, lcl, cl, ucl, n, L=L)

    def __getLimits(self, n, range_samples, L = 3):
        rBar = np.mean(range_samples)
        D3 = getD3(n, L)
        D4 = getD4(n, L)
        return D3 * rBar, rBar, D4 * rBar 

class SChart:
    def __init__(self, samples, L=3, title="S Chart"):
        self.samples = samples
        self.L = L
        self.title = title
        self.num_iterations = 1
        self.OC_samples = {}
        self.control_limits ={}
        self.__init_sChart(samples, L=L)
        self.finalLCL = self.get_final_control_limits()["LCL"]
        self.finalCL = self.get_final_control_limits()["CL"]
        self.finalUCL = self.get_final_control_limits()["UCL"]
        self.phase2_samples = []

    def show_charts(self):
        self.__sChart(self.samples, L=self.L)
    
    def show_phase2_chart(self):
        std_samples = np.std(self.phase2_samples, axis=1)
        plot(std_samples, self.finalLCL, self.finalCL, self.finalUCL, title=self.title)

    def get_num_iterations(self):
        return self.num_iterations
    
    def get_oc_samples(self):
        return self.OC_samples

    def get_control_limits(self):
        return self.control_limits

    def get_final_control_limits(self):
        return self.control_limits[f"Iteration {self.num_iterations}"]

    def summary(self):
        return f"Iterations: {self.num_iterations}\nControl Limits: {self.control_limits}\nOC Samples: {self.OC_samples}"

    def add_phase2_samples(self, samples):
        self.phase2_samples = samples
    
    def __sChart(self, samples, L = 3):
        n = len(samples[0])
        std_samples = np.std(samples, axis=1)
        lcl, cl, ucl = self.__getLimits(n, std_samples, L=L) 
        plot(std_samples, lcl, cl, ucl, title=self.title)
        self.__checkWithinLimits(std_samples, lcl, cl, ucl, L=L, n=n)

    def __init_sChart(self, samples, L = 3):
        n = len(samples[0])
        std_samples = np.std(samples, axis=1)
        lcl, cl, ucl = self.__getLimits(n, std_samples, L=L) 
        self.__init_checkWithinLimits(std_samples, lcl, cl, ucl, L=L, n=n)

    def __checkWithinLimits(self, samples, lcl, cl, ucl, n, L = 3):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples, L=L)
        if len(updated_samples) != len(samples):
            plot(updated_samples, lcl, cl, ucl, title=self.title)
            self.__checkWithinLimits(updated_samples, lcl, cl, ucl, n, L=L)
            
    def __init_checkWithinLimits(self, samples, lcl, cl, ucl, n, L = 3):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples, L=L)
        self.control_limits[f"Iteration {self.num_iterations}"] = {"LCL": lcl, "CL": cl, "UCL": ucl} 
        if len(updated_samples) != len(samples):
            self.OC_samples[f"Iteration {self.num_iterations}"] = OC_samples
            self.num_iterations += 1
            self.__init_checkWithinLimits(updated_samples, lcl, cl, ucl, n, L=L)

    def __getLimits(self, n, std_samples, L = 3):
        sBar = np.mean(std_samples)
        B3 = getB3(n, L)
        B4 = getB4(n, L)
        return B3 * sBar, sBar, B4 * sBar

class XBarChart:
    def __init__(self, samples, L=3, variation_type="Range", title="Xbar Chart"):
        self.variation_type = variation_type
        if self.variation_type == "Range":
            r = RChart(samples, L=L)
            self.std = r.finalCL
        else:
            s = SChart(samples, L=L)
            self.std = s.finalCL
        self.samples = samples
        self.L = L
        self.title = title
        self.num_iterations = 1
        self.OC_samples = {}
        self.control_limits ={}
        self.__init_xBarChart(samples, L=L)
        self.finalLCL = self.get_final_control_limits()["LCL"]
        self.finalCL = self.get_final_control_limits()["CL"]
        self.finalUCL = self.get_final_control_limits()["UCL"]
        self.phase2_samples = []

    def show_charts(self):
        self.__xBarChart(self.samples, L=self.L)
    
    def show_phase2_chart(self):
        mean_samples = np.mean(self.phase2_samples, axis=1)
        plot(mean_samples, self.finalLCL, self.finalCL, self.finalUCL, title=self.title)

    def get_num_iterations(self):
        return self.num_iterations
    
    def get_oc_samples(self):
        return self.OC_samples

    def get_control_limits(self):
        return self.control_limits

    def get_final_control_limits(self):
        return self.control_limits[f"Iteration {self.num_iterations}"]

    def summary(self):
        return f"Iterations: {self.num_iterations}\nControl Limits: {self.control_limits}\nOC Samples: {self.OC_samples}"

    def add_phase2_samples(self, samples):
        self.phase2_samples = samples
    
    def __xBarChart(self, samples, L = 3):
        n = len(samples[0])
        mean_samples = np.mean(samples, axis=1)
        lcl, cl, ucl = self.__getLimits(n, mean_samples, L=L) 
        plot(mean_samples, lcl, cl, ucl, title=self.title)
        self.__checkWithinLimits(mean_samples, lcl, cl, ucl, L=L, n=n)

    def __init_xBarChart(self, samples, L = 3):
        n = len(samples[0])
        mean_samples = np.mean(samples, axis=1)
        lcl, cl, ucl = self.__getLimits(n, mean_samples, L=L) 
        self.__init_checkWithinLimits(mean_samples, lcl, cl, ucl, L=L, n=n)

    def __checkWithinLimits(self, samples, lcl, cl, ucl, n, L = 3):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples, L=L)
        if len(updated_samples) != len(samples):
            plot(updated_samples, lcl, cl, ucl, title=self.title)
            self.__checkWithinLimits(updated_samples, lcl, cl, ucl, n, L=L)
            
    def __init_checkWithinLimits(self, samples, lcl, cl, ucl, n, L = 3):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples, L=L)
        self.control_limits[f"Iteration {self.num_iterations}"] = {"LCL": lcl, "CL": cl, "UCL": ucl} 
        if len(updated_samples) != len(samples):
            self.OC_samples[f"Iteration {self.num_iterations}"] = OC_samples
            self.num_iterations += 1
            self.__init_checkWithinLimits(updated_samples, lcl, cl, ucl, n, L=L)

    def __getLimits(self, n, mean_samples, L = 3):
        xDoubleBar = np.mean(mean_samples)
        if self.variation_type == "Range":
            A2 = getA2(n, L)
            return xDoubleBar - A2 * self.std, xDoubleBar, xDoubleBar + A2 * self.std
        else:
            A3 = getA3(n, L)
            return xDoubleBar - A3 * self.std, xDoubleBar, xDoubleBar + A3 * self.std
            




def general(samples, lcl, cl, ucl, title = None):
    plot(samples, lcl, cl, ucl, title=title)
    checkWithinLimits(samples, lcl, cl, ucl, title=title)

def xbarChart(samples, L = 3, title = None):
    lcl, ucl, cl = getXbarLimits(samples, L=L)
    plot(samples, lcl, cl, ucl, title=title)
    checkWithinLimitsXbar(samples, lcl, cl, ucl, L=L, title=title)

def plot(samples, lcl, cl, ucl, title = None):
    plt.axhline(lcl, color='black', label=f'LCL: {round(lcl, 4)}')
    plt.axhline(cl, color='blue', label=f'CL: {round(cl, 4)}')
    plt.axhline(ucl, color='black', label=f'UCL: {round(ucl, 4)}')
    plt.plot(samples)
    plt.legend()
    plt.title(title)
    plt.show()
    
def checkWithinLimits(samples, lcl, cl, ucl, title = None):
    updated_samples = samples[(samples > lcl) & (samples < ucl)]
    if len(updated_samples) != len(samples):
        plot(updated_samples, lcl, cl, ucl, title=f"Updated {title}")
        checkWithinLimits(updated_samples, lcl, cl, ucl, title=title)

def checkWithinLimitsXbar(samples, lcl, cl, ucl, L = 3, title = None):
    updated_samples = samples[(samples > lcl) & (samples < ucl)]
    lcl, cl, ucl = getXbarLimits(updated_samples, L=L)
    if len(updated_samples) != len(samples):
        plot(updated_samples, lcl, cl, ucl, title=f"Updated {title}")
        checkWithinLimitsXbar(updated_samples, lcl, cl, ucl, L=L, title=title)

def getXbarLimits(samples, L = 3):
    n = len(samples)
    sigma = np.std(samples)
    cl = np.mean(samples)
    lcl = cl - L * sigma / sqrt(n)
    ucl = cl + L * sigma / sqrt(n)
    return (lcl, cl, ucl)