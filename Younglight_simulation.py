#!/home/guoxiaotong/anaconda3/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 - 2025 by Guo Xiaotong. All rights reserved.
# @Time    : 2025/11/7 15:47
# @Author  : Guo Xiaotong
# @FileName: Younglight_simulation.py
# @Software: PyCharm
"""------------------导入第三方库--------------------"""
import numpy as np
import os
import astropy.units as u
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
"""------------------编写类与函数--------------------"""
def wavelength_to_rgb(wavelength, gamma=0.8):
    """CIE标准波长转RGB算法实现"""
    if not 390 <= wavelength <= 760:
        #raise ValueError("波长需在390-760nm范围内")
        print("波长不在可见光波长390-760nm范围内，需要重新输入波长")
        return tuple((0, 0, 0))


    # 波长分段线性插值
    if wavelength < 450:
        r = -(wavelength - 450) / (450 - 380)
        g, b = 0, 1
    elif wavelength < 500:
        r, g = 0, (wavelength - 450) / (500 - 450)
        b = 1
    elif wavelength < 520:
        r, b = 0, -(wavelength - 520) / (520 - 500)
        g = 1
    elif wavelength < 590:
        r = (wavelength - 520) / (590 - 520)
        g, b = 1, 0
    elif wavelength < 655:
        r, g = 1, -(wavelength - 655) / (655 - 590)
        b = 0
    else:
        r, g, b = 1, 0, 0

    # 优化后的强度校正
    if wavelength < 430:
        factor = 0.3 + 0.7 * (wavelength - 390) / (430 - 390)
    elif wavelength < 655:
        factor = 1.0
    elif wavelength <= 690:  # 645-680nm二次曲线过渡
        x = (wavelength - 655) / (690 - 655)
        factor = 1.0 - 0.5 * x ** 2
    else:  # 680nm以上保持基础亮度
        factor = 0.5

    # 红色通道保护（660nm以上）
    if wavelength > 660:
        r = max(r, 0.8)  # 确保红色通道最低亮度

    def adjust(color):
        return np.clip((color * factor) ** gamma, 0, 255)

    return tuple(map(float,map(adjust, (r, g, b))))
class Interference:
    def __init__(self,wavelength=550,
                 delta_wavelength=0,
                 S_width=0.001,
                 S1_width=0.001,
                 S2_width=0.001,
                 R=50,
                 d=0.2,
                 r0= 1000,
                 I0= 1):
        self.wavelength = wavelength * u.nm
        self.delta_wavelength = delta_wavelength * u.nm
        self.S_width = S_width * u.mm
        self.S1_width = S1_width * u.mm
        self.S2_width = S2_width * u.mm
        self.R = R * u.mm
        self.d = d * u.mm
        self.r0 = r0 * u.mm
        self.I0 = I0

    def ideal_OPD(self,y):
        if hasattr(y, 'unit'):
            y_coordinate = y.to(u.mm)
        else:
            y_coordinate = y * u.mm
        r_1 = np.sqrt(self.r0 ** 2 + (y_coordinate - self.d / 2) ** 2)
        r_2 = np.sqrt(self.r0 ** 2 + (y_coordinate + self.d / 2) ** 2)
        OPD_delta = r_2 - r_1
        return OPD_delta
    def center_wavelength_color(self):
        wavelength_to_color = np.array(wavelength_to_rgb(self.wavelength.to(u.nm).value))
        return wavelength_to_color

    def Amplitude_ratio(self,y,E_1_to_E_2_ratio=1):
        OPD_delta = self.ideal_OPD(y)
        I = E_1_to_E_2_ratio**2+1+2*E_1_to_E_2_ratio*np.cos((2*np.pi*OPD_delta/self.wavelength.to(u.mm)).value)
        I /= 2*(1+ E_1_to_E_2_ratio**2)

        wavelength_to_color = self.center_wavelength_color()
        rgb = np.array([color * I for color in wavelength_to_color]).T
        #print(rgb)
        return I, rgb
    def Slit_scale(self,y):

        OPD_delta = self.ideal_OPD(y)
        if hasattr(y, 'unit'):
            y_coordinate = y.to(u.mm)
        else:
            y_coordinate = y * u.mm
        u1= self.S1_width * y_coordinate/ self.wavelength/self.r0
        u2= self.S2_width * y_coordinate/ self.wavelength/self.r0
        I =  0.5*(1 + np.sinc(u1.to(u.one).value)*np.sinc(u2.to(u.one).value)*np.cos((2*np.pi*OPD_delta/self.wavelength.to(u.mm)).value))
        wavelength_to_color = self.center_wavelength_color()
        rgb = np.array([color * I for color in wavelength_to_color]).T
        return I,rgb


    def Source_scale(self,y):
        OPD_delta = self.ideal_OPD(y)
        I = 0.5*(1+np.sinc((self.d*self.S_width/self.wavelength/self.R).to(u.one).value)*
                         np.cos(2 * np.pi * (OPD_delta/self.wavelength).to(u.one).value))
        wavelength_to_color = self.center_wavelength_color()
        rgb = np.array([color * I for color in wavelength_to_color]).T
        return I, rgb
    def Non_single_color_of_source(self,y):
        OPD_delta = self.ideal_OPD(y)
        wavelength_to_color = self.center_wavelength_color()

        if self.delta_wavelength > 0*u.nm:

            sigma = self.delta_wavelength/(2*np.sqrt(2*np.log(2)))

            C_delta = (np.exp(-2*np.pi**2 *((sigma * OPD_delta/self.wavelength**2).to(u.one).value)**2 ) *
                       np.cos(2 * np.pi * (OPD_delta/self.wavelength.to(u.mm)).value))
            I_non_single_wavelength = 0.5* (1 + C_delta)
            rgb = np.array([color * I_non_single_wavelength for color in wavelength_to_color]).T

            return I_non_single_wavelength,rgb
        elif self.delta_wavelength == 0*u.nm:
            I_single_wavelength = 0.5*(1+np.cos(2 * np.pi * (OPD_delta/self.wavelength.to(u.mm)).value))
            rgb = np.array([color * I_single_wavelength for color in wavelength_to_color]).T
            return I_single_wavelength,rgb
        else:
            print('Delta lambda is error,please check.')

    def actual_interference_RGB(self,y,spec_shape='gaussian'):
        OPD_delta = self.ideal_OPD(y)
        if hasattr(y, 'unit'):
            y_coordinate = y.to(u.mm)
        else:
            y_coordinate = y * u.mm

        u1 = self.S1_width * y_coordinate / self.wavelength / self.r0
        u2 = self.S2_width * y_coordinate / self.wavelength / self.r0
        V_slits = np.sinc(u1.to(u.one).value) * np.sinc(u2.to(u.one).value)

        u_src = self.d * self.S_width / self.wavelength / self.R
        V_src = np.sinc(u_src.to(u.one).value)

        V_total = V_slits * V_src


        num_lambda = 500
        if spec_shape == 'gaussian':
            sigma = self.delta_wavelength / (2 * np.sqrt(2 * np.log(2)))
            lams = np.linspace(self.wavelength - 5 * sigma, self.wavelength + 5 * sigma, num_lambda + 1)
            #print(lams)
            I_0 = norm.pdf(x=lams, loc=self.wavelength, scale=sigma)

        elif spec_shape == 'uniform':
            sigma = self.delta_wavelength / 2
            lams = np.linspace(self.wavelength - sigma, self.wavelength + sigma, num_lambda + 1)
            I_0 = np.full(num_lambda+1,1/self.delta_wavelength.to(u.nm).value)
            #print(I_0)
        elif spec_shape == 'lorentz':
            #未完善
            sigma = self.delta_wavelength / 2
            lams = np.linspace(self.wavelength - sigma, self.wavelength + sigma, num_lambda + 1)
            I_0 = norm.pdf(x=lams, loc=self.wavelength, scale=sigma)
        else:
            print('Spec_shape is error,please check.')
        #print(I_0)
        weight = []
        rgb = []
        I_wavelength = []


        for i in range(num_lambda):
            wavelength = (lams[i] + lams[i + 1]) / 2
            single_weight = np.exp(-0.5 * ((wavelength.to(u.nm).value - 550) / 80) ** 2) + 0.2
            weight.append(single_weight)

            wavelength_to_color = wavelength_to_rgb(wavelength.to(u.nm).value)
            rgb.append(list(wavelength_to_color))


            #I_0 = norm.pdf(x=wavelength, loc=self.wavelength, scale=sigma) * ((lams[i + 1] - lams[i]).to(u.nm)).value
            I_temp = (I_0[i]+I_0[i+1])/2 * ((lams[i + 1] - lams[i]).to(u.nm)).value

            #I_single_wavelength = 0.5 *I_0* (1+ V_total * np.cos(2 * np.pi * (OPD_delta/self.wavelength.to(u.mm)).value))
            I_single_wavelength = 0.5 * I_temp * (
                        1 + V_total * np.cos(2 * np.pi * (OPD_delta / wavelength.to(u.mm)).value))

            I_wavelength.append(I_single_wavelength)


        RGB_wavelength = np.einsum('wy,wj->yj',  I_wavelength, rgb)
        #RGB = np.array([RGB_wavelength.T[0],RGB_wavelength.T[1],RGB_wavelength.T[2]])
        #RGB_wavelength = RGB.T / RGB.max()
        RGB_wavelength /= RGB_wavelength.max()
        #print(RGB_wavelength)

        return np.sum(np.array(I_wavelength),axis=0),RGB_wavelength


def simulation_figures():
    wavelength = 575
    delta_wavelength = 370
    d = 0.4
    r0 = 1000
    R = 300
    b = 0.01
    b1 = 0.003
    b2 = 0.003

    duoble_silts = Interference(wavelength=wavelength, delta_wavelength=delta_wavelength, d=d, r0=r0, R=R,
                                S_width=b, S1_width=b1, S2_width=b2)

    x_lim = 10
    x = np.linspace(-x_lim, x_lim, 1001)
    #y, RGB_colors = duoble_silts.Amplitude_ratio(x,E_1_to_E_2_ratio=0.1)
    #y, RGB_colors = duoble_silts.Slit_scale(x)
    #y, RGB_colors = duoble_silts.Source_scale(x)
    #y, RGB_colors = duoble_silts.Non_single_color_of_source(x)
    y, RGB_colors = duoble_silts.actual_interference_RGB(x,spec_shape='uniform')


    fig, ax = plt.subplots(1, 1, figsize=(7.3, 5.5))
    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(x, y, color='r')
    ax.plot([-x_lim, x_lim], [0.5, 0.5], linestyle='--', linewidth=0.5, color='k')
    ax.text(0.15, 0.8, '$I$', color='k', fontsize=18)
    # ax.set_xticks([0], ['$\\Delta y$=0'], fontsize=15)
    ax.set_yticks([0.5, 1], ['$\\bar{\\mathrm{I}}$', '2$\\bar{\\mathrm{I}}$'], fontsize=12)
    ax.set_xlim([-x_lim, x_lim])
    ax.set_ylim([0, 1.25])
    ax.set_xlabel('Screen Position (mm)', fontsize=15)
    ax.set_ylabel('Normalized Intensity', fontsize=15)
    #ax.legend('2')

    axlow = fig.add_axes([0.125, 0.785, 0.775, 0.1])
    axlow.set_xticks([])
    axlow.set_yticks([])
    # axlow.set_xlim([-20,20])
    axlow.set_ylim([0, int(len(y) / 11)])
    # for i in range(len(I_sum_interference)):
    # axlow.plot([y[i],y[i]],[0,1],color='k',alpha = 1-I_sum_interference[i]/2.5,linewidth=0.3)
    """
    for yi, inten in zip(y, I_sum_interference):  # 抽稀加速
        alpha = 1 - inten / 2.5
        axlow.plot([yi, yi], [0, 1], c='k', alpha=alpha, lw=0.165)
    """
    fig_interference = []
    for i in range(int(len(RGB_colors) / 11)):
        fig_interference.append(np.clip(RGB_colors, 0, 1))

    image = axlow.imshow(fig_interference, vmin=0, vmax=1)
    # cb = fig.colorbar(image, ax=ax, fraction=0.07, pad=0.07)
    # ax.legend(loc='upper left', fontsize=15)
    if not os.path.exists('output'):
        os.makedirs('output')
    fig.savefig(f'output/simulation.pdf')
    plt.show()

def main():
    simulation_figures()
    pass


"""------------------程序运行区--------------------"""
if __name__ == '__main__':
    main()
