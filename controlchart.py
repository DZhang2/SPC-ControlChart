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
        self.__init_rChart(samples, )
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
    
    def __rChart(self, samples):
        n = len(samples[0])
        range_samples = np.ptp(samples, axis=1) 
        lcl, cl, ucl = self.__getLimits(n, range_samples) 
        plot(range_samples, lcl, cl, ucl, title=self.title)
        self.__checkWithinLimits(range_samples, lcl, cl, ucl, n=n)

    def __init_rChart(self, samples):
        n = len(samples[0])
        range_samples = np.ptp(samples, axis=1) 
        lcl, cl, ucl = self.__getLimits(n, range_samples) 
        self.__init_checkWithinLimits(range_samples, lcl, cl, ucl, n=n)

    def __checkWithinLimits(self, samples, lcl, cl, ucl, n):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples)
        if len(updated_samples) != len(samples):
            plot(updated_samples, lcl, cl, ucl, title=self.title)
            self.__checkWithinLimits(updated_samples, lcl, cl, ucl, n)
            
    def __init_checkWithinLimits(self, samples, lcl, cl, ucl, n):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples)
        self.control_limits[f"Iteration {self.num_iterations}"] = {"LCL": lcl, "CL": cl, "UCL": ucl} 
        if len(updated_samples) != len(samples):
            self.OC_samples[f"Iteration {self.num_iterations}"] = OC_samples
            self.num_iterations += 1
            self.__init_checkWithinLimits(updated_samples, lcl, cl, ucl, n)

    def __getLimits(self, n, range_samples):
        rBar = np.mean(range_samples)
        D3 = getD3(n, self.L)
        D4 = getD4(n, self.L)
        return D3 * rBar, rBar, D4 * rBar 

class SChart:
    def __init__(self, samples, L=3, title="S Chart"):
        self.samples = samples
        self.L = L
        self.title = title
        self.num_iterations = 1
        self.OC_samples = {}
        self.control_limits ={}
        self.__init_sChart(samples)
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
    
    def __sChart(self, samples):
        n = len(samples[0])
        std_samples = np.std(samples, axis=1)
        lcl, cl, ucl = self.__getLimits(n, std_samples) 
        plot(std_samples, lcl, cl, ucl, title=self.title)
        self.__checkWithinLimits(std_samples, lcl, cl, ucl, n=n)

    def __init_sChart(self, samples):
        n = len(samples[0])
        std_samples = np.std(samples, axis=1)
        lcl, cl, ucl = self.__getLimits(n, std_samples) 
        self.__init_checkWithinLimits(std_samples, lcl, cl, ucl, n=n)

    def __checkWithinLimits(self, samples, lcl, cl, ucl, n):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples)
        if len(updated_samples) != len(samples):
            plot(updated_samples, lcl, cl, ucl, title=self.title)
            self.__checkWithinLimits(updated_samples, lcl, cl, ucl, n)
            
    def __init_checkWithinLimits(self, samples, lcl, cl, ucl, n):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples)
        self.control_limits[f"Iteration {self.num_iterations}"] = {"LCL": lcl, "CL": cl, "UCL": ucl} 
        if len(updated_samples) != len(samples):
            self.OC_samples[f"Iteration {self.num_iterations}"] = OC_samples
            self.num_iterations += 1
            self.__init_checkWithinLimits(updated_samples, lcl, cl, ucl, n)

    def __getLimits(self, n, std_samples):
        sBar = np.mean(std_samples)
        B3 = getB3(n, self.L)
        B4 = getB4(n, self.L)
        return B3 * sBar, sBar, B4 * sBar

class XBarChart:
    def __init__(self, samples, L=3, variation_type="Range", title="Xbar Chart"):
        self.variation_type = variation_type
        if self.variation_type == "Range":
            r = RChart(samples)
            self.std = r.finalCL
        else:
            s = SChart(samples)
            self.std = s.finalCL
        self.samples = samples
        self.L = L
        self.title = title
        self.num_iterations = 1
        self.OC_samples = {}
        self.control_limits ={}
        self.__init_xBarChart(samples)
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
    
    def __xBarChart(self, samples):
        n = len(samples[0])
        mean_samples = np.mean(samples, axis=1)
        lcl, cl, ucl = self.__getLimits(n, mean_samples) 
        plot(mean_samples, lcl, cl, ucl, title=self.title)
        self.__checkWithinLimits(mean_samples, lcl, cl, ucl, n=n)

    def __init_xBarChart(self, samples):
        n = len(samples[0])
        mean_samples = np.mean(samples, axis=1)
        lcl, cl, ucl = self.__getLimits(n, mean_samples) 
        self.__init_checkWithinLimits(mean_samples, lcl, cl, ucl, n=n)

    def __checkWithinLimits(self, samples, lcl, cl, ucl, n):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples)
        if len(updated_samples) != len(samples):
            plot(updated_samples, lcl, cl, ucl, title=self.title)
            self.__checkWithinLimits(updated_samples, lcl, cl, ucl, n)
            
    def __init_checkWithinLimits(self, samples, lcl, cl, ucl, n):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(n, updated_samples)
        self.control_limits[f"Iteration {self.num_iterations}"] = {"LCL": lcl, "CL": cl, "UCL": ucl} 
        if len(updated_samples) != len(samples):
            self.OC_samples[f"Iteration {self.num_iterations}"] = OC_samples
            self.num_iterations += 1
            self.__init_checkWithinLimits(updated_samples, lcl, cl, ucl, n)

    def __getLimits(self, n, mean_samples):
        xDoubleBar = np.mean(mean_samples)
        if self.variation_type == "Range":
            A2 = getA2(n, self.L)
            return xDoubleBar - A2 * self.std, xDoubleBar, xDoubleBar + A2 * self.std
        else:
            A3 = getA3(n, self.L)
            return xDoubleBar - A3 * self.std, xDoubleBar, xDoubleBar + A3 * self.std

class pChart:
    def __init__(self, samples, n, L=3, title="P Chart"):
        self.samples = samples
        self.n = n
        self.L = L
        self.title = title
        self.num_iterations = 1
        self.OC_samples = {}
        self.control_limits ={}
        self.__init_pChart(samples)
        self.finalLCL = self.get_final_control_limits()["LCL"]
        self.finalCL = self.get_final_control_limits()["CL"]
        self.finalUCL = self.get_final_control_limits()["UCL"]
        self.phase2_samples = []

    def show_charts(self):
        self.__pChart(self.samples, L=self.L, title=self.title)
    
    def show_phase2_chart(self):
        p_samples = self.phase2_samples/self.n
        plot(p_samples, self.finalLCL, self.finalCL, self.finalUCL, title=self.title)

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
    
    def __pChart(self, samples):
        p_samples = samples/self.n
        lcl, cl, ucl = self.__getLimits(p_samples) 
        plot(p_samples, lcl, cl, ucl, title=self.title)
        self.__checkWithinLimits(p_samples, lcl, cl, ucl)

    def __init_pChart(self, samples):
        p_samples = samples/self.n 
        lcl, cl, ucl = self.__getLimits(p_samples) 
        self.__init_checkWithinLimits(p_samples, lcl, cl, ucl)

    def __checkWithinLimits(self, samples, lcl, cl, ucl):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(updated_samples)
        if len(updated_samples) != len(samples):
            plot(updated_samples, lcl, cl, ucl, title=self.title)
            self.__checkWithinLimits(updated_samples, lcl, cl, ucl)
            
    def __init_checkWithinLimits(self, samples, lcl, cl, ucl):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(updated_samples)
        self.control_limits[f"Iteration {self.num_iterations}"] = {"LCL": lcl, "CL": cl, "UCL": ucl} 
        if len(updated_samples) != len(samples):
            self.OC_samples[f"Iteration {self.num_iterations}"] = OC_samples
            self.num_iterations += 1
            self.__init_checkWithinLimits(updated_samples, lcl, cl, ucl)

    def __getLimits(self, p_samples):
        pBar = np.mean(p_samples)
        cl = pBar
        lcl = max(0, pBar - self.L * sqrt(pBar*(1-pBar)/self.n))
        ucl = pBar + self.L * sqrt(pBar*(1-pBar)/self.n) 
        return lcl, cl, ucl

class npChart:
    def __init__(self, samples, n, L=3, title="NP Chart"):
        self.samples = samples
        self.n = n
        self.L = L
        self.title = title
        self.num_iterations = 1
        self.OC_samples = {}
        self.control_limits ={}
        self.__init_pChart(samples)
        self.finalLCL = self.get_final_control_limits()["LCL"]
        self.finalCL = self.get_final_control_limits()["CL"]
        self.finalUCL = self.get_final_control_limits()["UCL"]
        self.phase2_samples = []

    def show_charts(self):
        self.__npChart(self.samples, L=self.L, title=self.title)
    
    def show_phase2_chart(self):
        plot(self.phase2_samples, self.finalLCL, self.finalCL, self.finalUCL, title=self.title)

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
    
    def __npChart(self, samples):
        p_samples = samples/self.n
        lcl, cl, ucl = self.__getLimits(p_samples) 
        plot(n * p_samples, lcl, cl, ucl, title=self.title)
        self.__checkWithinLimits(p_samples, lcl, cl, ucl)

    def __init_npChart(self, samples):
        p_samples = samples/self.n
        lcl, cl, ucl = self.__getLimits(p_samples) 
        self.__init_checkWithinLimits(p_samples, lcl, cl, ucl)

    def __checkWithinLimits(self, samples, lcl, cl, ucl):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(updated_samples)
        if len(updated_samples) != len(samples):
            plot(n * updated_samples, lcl, cl, ucl, title=self.title)
            self.__checkWithinLimits(updated_samples, lcl, cl, ucl)
            
    def __init_checkWithinLimits(self, samples, lcl, cl, ucl):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(updated_samples)
        self.control_limits[f"Iteration {self.num_iterations}"] = {"LCL": lcl, "CL": cl, "UCL": ucl} 
        if len(updated_samples) != len(samples):
            self.OC_samples[f"Iteration {self.num_iterations}"] = OC_samples
            self.num_iterations += 1
            self.__init_checkWithinLimits(updated_samples, lcl, cl, ucl)

    def __getLimits(self, p_samples):
        pBar = np.mean(p_samples)
        cl = n * pBar
        lcl = max(0, n * pBar - self.L * sqrt(n * pBar * (1 - pBar)))
        ucl = n * pBar + self.L * sqrt(n * pBar * (1 - pBar)) 
        return lcl, cl, ucl

class cChart:
    def __init__(self, samples, n, L=3, title="C Chart"):
        self.samples = samples
        self.n = n
        self.L = L
        self.title = title
        self.num_iterations = 1
        self.OC_samples = {}
        self.control_limits ={}
        self.__init_pChart(samples)
        self.finalLCL = self.get_final_control_limits()["LCL"]
        self.finalCL = self.get_final_control_limits()["CL"]
        self.finalUCL = self.get_final_control_limits()["UCL"]
        self.phase2_samples = []

    def show_charts(self):
        self.__cChart(self.samples, L=self.L, title=self.title)
    
    def show_phase2_chart(self):
        plot(self.phase2_samples, self.finalLCL, self.finalCL, self.finalUCL, title=self.title)

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
    
    def __cChart(self, samples):
        lcl, cl, ucl = self.__getLimits(samples) 
        plot(samples, lcl, cl, ucl, title=self.title)
        self.__checkWithinLimits(samples, lcl, cl, ucl)

    def __init_cChart(self, samples):
        lcl, cl, ucl = self.__getLimits(samples) 
        self.__init_checkWithinLimits(samples, lcl, cl, ucl)

    def __checkWithinLimits(self, samples, lcl, cl, ucl):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(updated_samples)
        if len(updated_samples) != len(samples):
            plot(updated_samples, lcl, cl, ucl, title=self.title)
            self.__checkWithinLimits(updated_samples, lcl, cl, ucl)
            
    def __init_checkWithinLimits(self, samples, lcl, cl, ucl):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(updated_samples)
        self.control_limits[f"Iteration {self.num_iterations}"] = {"LCL": lcl, "CL": cl, "UCL": ucl} 
        if len(updated_samples) != len(samples):
            self.OC_samples[f"Iteration {self.num_iterations}"] = OC_samples
            self.num_iterations += 1
            self.__init_checkWithinLimits(updated_samples, lcl, cl, ucl)

    def __getLimits(self, c_samples):
        cBar = np.mean(c_samples) 
        cl = cBar
        lcl = max(0, cBar - self.L * sqrt(c))
        ucl = cBar + self.L * sqrt(c)
        return lcl, cl, ucl
    
class cChart:
    def __init__(self, samples, n, L=3, title="U Chart"):
        self.samples = samples
        self.n = n
        self.L = L
        self.title = title
        self.num_iterations = 1
        self.OC_samples = {}
        self.control_limits ={}
        self.__init_pChart(samples)
        self.finalLCL = self.get_final_control_limits()["LCL"]
        self.finalCL = self.get_final_control_limits()["CL"]
        self.finalUCL = self.get_final_control_limits()["UCL"]
        self.phase2_samples = []

    def show_charts(self):
        self.__cChart(self.samples, L=self.L, title=self.title)
    
    def show_phase2_chart(self):
        plot(self.phase2_samples, self.finalLCL, self.finalCL, self.finalUCL, title=self.title)

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
    
    def __uChart(self, samples):
        lcl, cl, ucl = self.__getLimits(samples) 
        plot(n * samples, lcl, cl, ucl, title=self.title)
        self.__checkWithinLimits(samples, lcl, cl, ucl)

    def __init_uChart(self, samples):
        lcl, cl, ucl = self.__getLimits(samples) 
        self.__init_checkWithinLimits(samples, lcl, cl, ucl)

    def __checkWithinLimits(self, samples, lcl, cl, ucl):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(updated_samples)
        if len(updated_samples) != len(samples):
            plot(n * updated_samples, lcl, cl, ucl, title=self.title)
            self.__checkWithinLimits(updated_samples, lcl, cl, ucl)
            
    def __init_checkWithinLimits(self, samples, lcl, cl, ucl):
        updated_samples = samples[(samples > lcl) & (samples < ucl)]
        OC_samples = samples[(samples <= lcl) | (samples >= ucl)]
        lcl, cl, ucl = self.__getLimits(updated_samples)
        self.control_limits[f"Iteration {self.num_iterations}"] = {"LCL": lcl, "CL": cl, "UCL": ucl} 
        if len(updated_samples) != len(samples):
            self.OC_samples[f"Iteration {self.num_iterations}"] = OC_samples
            self.num_iterations += 1
            self.__init_checkWithinLimits(updated_samples, lcl, cl, ucl)

    def __getLimits(self, c_samples):
        cBar = np.mean(c_samples) 
        cl = n * cBar
        lcl = max(0, n * cBar - self.L * sqrt(n * c))
        ucl = n * cBar + self.L * sqrt(n * c)
        return lcl, cl, ucl

def general(samples, lcl, cl, ucl, title = None):
    plot(samples, lcl, cl, ucl, title=title)

def plot(samples, lcl, cl, ucl, title = None):
    plt.axhline(lcl, color='black', label=f'LCL: {round(lcl, 4)}')
    plt.axhline(cl, color='blue', label=f'CL: {round(cl, 4)}')
    plt.axhline(ucl, color='black', label=f'UCL: {round(ucl, 4)}')
    plt.plot(samples)
    plt.legend()
    plt.title(title)
    plt.show()