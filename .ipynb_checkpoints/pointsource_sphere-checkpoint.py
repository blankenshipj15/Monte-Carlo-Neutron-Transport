# -*- coding: utf-8 -*-
"""pointsource_sphere

Description of this file:
    - Solution of the point source within a sphere
    - The heterogenous sphere is divided into multiple shells
    - Only the absorption reaction is considered here for simplicity
    - It is assumed that the source does not absrob the neutrons
    - Although the source radius is defined, it is over-written with zero

Created on Sat May 17 18:20:00 2025 @author: Dan Kotlyar
Last updated on Mon May 19 11:30:00 2025 @author: Dan Kotlyar

email: dan.kotlyar@me.gatech.edu


"""

import time
import matplotlib.pyplot as plt
# from matplotlib import rcParams
import numpy as np
# rcParams['figure.dpi'] = 300

FONT_SIZE = 16
MARKER_SIZE = 6

ALLOWED_SCHEMES = ['st', 'dt', 'analytic']  # surface / delta tracking


class PointSourceInSphere():
    """MC solution for a point source positioned in an heterogenous sphere

    Parameters
    ----------
    nMC : int
        number of repetitive monte carlo simulations
    S0 : int
        number of source neutrons
    R : float
        radius of the sphere in cm
    sigT : array
        total cross section array for all the different regions


    Attributes
    ----------
    inps : dict
        container with inputs
    resAN : dict
        container with analytic results
    resST : dict
        container with surface tracking results
    resDT : dict
        container with delta tracking results
    times : dict
        container with execution times

    Raises
    -------
    TypeError
        if nMC is not int
    ValueError
        if nMC is not positive  

    """

    def __init__(self, nMC, S0, R, sigT):
        
        # check errors (example)
        if not isinstance(nMC, int):
            raise TypeError("nMC must be integer and not {}".format(nMC))
        if nMC <= 0:
            raise ValueError("nMC must be positive int and not {}".format(nMC))
        
        self.resST = {}  # surface tracking results
        self.resDT = {}  # delta tracking
        self.resAN = {}  # analytic results
        self.times = {}  # solution times
        
        nreg = len(sigT)
        vol = (4/3) * np.pi * R**3 / nreg  # single volume shell
        # define equal-volume radii
        radii = np.zeros(nreg)
        radii[0] = (vol*3/(4*np.pi))**(1/3)
        for i in range(1, nreg):
            radii[i] = (radii[i-1]**3 + vol*3/(4*np.pi))**(1/3)

        # save inputs dict
        self.inps = {'R': R, 'nMC': nMC, 'sigT': sigT, 'radii': radii, 
                     'nreg': nreg, 'S0': S0}
        


    def Solve(self, scheme):
        """MC solution using a specific random walk scheme
        
        Parameters
        ----------
        scheme : str
            neutron tracking scheme ['st', 'dt', 'analytic']

        """
        
        nMC = self.inps['nMC']
        nreg = self.inps['nreg']
        
        if scheme == 'analytic':
            self._Analytic()
            
        elif scheme.lower() == 'st':
            start_time = time.time()
            flux = np.zeros((nreg, nMC))
            leakage = np.zeros(nMC)
            for i in range(nMC):
                flux[:, i], leakage[i] = self._SolveST()  
            self.resST = {'flx': np.mean(flux, 1), 
                          'errflx': np.std(flux, 1),
                          'leakage': np.mean(leakage), 
                          'errleakage': np.std(leakage)}
            self.times['ST'] = time.time() - start_time
            
        elif scheme.lower() == 'dt':
            start_time = time.time()
            flux = np.zeros((nreg, nMC))
            leakage = np.zeros(nMC)
            for i in range(nMC):
                flux[:, i], leakage[i] = self._SolveDT()  
            self.resDT = {'flx': np.mean(flux, 1), 
                          'errflx': np.std(flux, 1),
                          'leakage': np.mean(leakage), 
                          'errleakage': np.std(leakage)}  
            self.times['DT'] = time.time() - start_time
        else:
            raise ValueError('scheme {} does not exist in {}'
                             .format(scheme,ALLOWED_SCHEMES))


    def _Analytic(self):
        """Analytic solution"""
        
        nreg = self.inps['nreg']
        radii = self.inps['radii']
        sigT = self.inps['sigT']
        
        # loop over all regions
        I0 = self.inps['S0']
        Ix_I0 = np.full(nreg, I0, dtype=float)
        for j in range(nreg):

            if j > 0:
                dr = radii[j] - radii[j-1]
                Ix_I0[j] = Ix_I0[j-1]*np.exp(-sigT[j]*dr)
            else:
                dr = radii[j] 
                Ix_I0[j] = I0*np.exp(-sigT[j]*dr)
                
        leakage = Ix_I0[-1] / I0
        self.resAN = {'flx': Ix_I0, 'leakage': leakage}


    def _SolveST(self):
        """MC solution using ray tracking"""
        
        nreg = self.inps['nreg']
        radii = self.inps['radii']
        sigT = self.inps['sigT']
        S0 = self.inps['S0']
        
        # reset neutron flux
        flux = np.zeros(nreg)
                
        # sample particles position (R, theta, phi)
        xipos = np.random.rand(S0, 3)
        
        # Particles are "born" in the most-inner sphere (R,theta,phi)
        Rin = 0.0  # fake radius (just to show how a position is sampled)
        R0 = Rin * xipos[:,0]**(1/3)          # sample radius
        theta = np.arccos( 1- 2*xipos[: , 1]) # theta angle   
        phi = 2*np.pi*xipos[: , 2]            # phi angle

        # Sample directional flight of neutrons
        xidir = np.random.rand(S0, 2)
        wtheta = np.arccos( 1- 2*xidir[: , 0])  # theta
        wphi = 2*np.pi*xipos[: , 1]             # phi angle
        
        for i in range(S0):
        
            # (X0,Y0,Z0) - before collision
            x0 = R0[i] * np.sin(theta[i]) * np.cos(phi[i])
            y0 = R0[i] * np.sin(theta[i]) * np.sin(phi[i])
            z0 = R0[i] * np.cos(theta[i])
            
            # (WX,WY,WZ) - direction
            wx = np.sin(wtheta[i]) * np.cos(wphi[i])
            wy = np.sin(wtheta[i]) * np.sin(wphi[i])
            wz = np.cos(wtheta[i])
                 
            # Solve the following inequety and find the intersection surfaces
            # Lx^2+Ly^2+Lz^2 = Ri^2 (eq.1)
            # Lx=x0+Si*wx ;Ly=y0+Si*wy; Lz=z0+Si*wz (eq.2)
            
            for j in range(nreg): # loop over all possible intersections
                # Sample collision's position
                xi = np.random.rand() # random number from Uniform [0,1]
                Si = -np.log(xi)/sigT[j]
                
                x1= x0 + Si*wx
                y1= y0 + Si*wy
                z1= z0 + Si*wz
                
                if  x1**2 + y1**2 + z1**2 < radii[j]**2:
                    # neutron is absorbed in this shell
                    break
                elif j == nreg-1:
                    # particle escaped from the system
                    dist = radii[j] - radii[j-1]  
                    flux[j] += 1  # score the neutron
                    break
                    
                else:
                    if j > 0:
                        dist = radii[j] - radii[j-1]
                    else:
                        dist = radii[j]
                    flux[j] += 1  # score the neutron
                    # update the position of the neutron
                    x0=x0+dist*wx
                    y0=y0+dist*wy
                    z0=z0+dist*wz
                    
        # get relative leakage fraction
        leakage = flux[-1] / S0
        
        return flux, leakage
                    
 
    def _SolveDT(self):
        """MC solution using delta tracking"""
        
        nreg = self.inps['nreg']
        radii = self.inps['radii']
        sigT = self.inps['sigT']
        S0 = self.inps['S0']
        radii2 = radii**2  # radii squared
        # majorant cross section
        sigMaj = max(sigT)
        
        # reset neutron flux
        flux = np.zeros(nreg)
                
        # sample particles position (R, theta, phi)
        xipos = np.random.rand(S0, 3)
        
        # Particles are "born" in the most-inner sphere (R,teta,phi)
        Rin = 0.0  # fake radius (just to show how a position is sampled)
        R0 = COMPLETE       # sample radius
        theta = np.arccos( 1-2*xipos[:, 1] )  # teta angle   
        phi = 2*np.pi*xipos[:, 2]             # phi angle

        # Sample directional flight of neutrons
        xidir = np.random.rand(S0, 2)
        wtheta = COMPLETE  # teta
        wphi = 2*np.pi*xidir[:, 1]            # phi angle
        
        for i in range(S0):
        
            # (X0,Y0,Z0) - before collision
            x0 = COMPLETE
            y0 = R0[i] * np.sin(theta[i]) * np.sin(phi[i])
            z0 = R0[i] * np.cos(theta[i])
            
            # (WX,WY,WZ) - direction
            wx = COMPLETE
            wy = np.sin(wtheta[i]) * np.sin(wphi[i])
            wz = np.cos(wtheta[i])
            
            idx1 = 0
            
            while 1:
            
                xi = np.random.rand() # random number from Uniform [0,1]
                Si = COMPLETE
            
                x1=COMPLETE
                y1=COMPLETE
                z1=COMPLETE
                
                R2 = x1**2+y1**2+z1**2
                
                if COMPLETE >= radii2[-1]:
                    # neutron leaked
                    COMPLETE  # neutron crossed all shells
                    break
                else:
                    idx1 = COMPLETE
                    # woodcock rejection method
                    if np.random.rand() < COMPLETE:
                        # real collision
                        COMPLETE  # score 
                        break
                    else:
                        # virtual collision
                        x0 = COMPLETE
                        y0 = COMPLETE
                        z0 = COMPLETE
                        
                
        # get relative leakage fraction
        leakage = flux[-1] / S0
        
        return flux, leakage                

                
    def PlotFluxes(self, scheme):
        """plot the fluxes"""
    
        if not isinstance(scheme, str):
            raise TypeError("scheme must be str and not {}".format(scheme))
        radii = self.inps['radii']
        ylabel = "Flux [#]"
        xlabel = "Distance from center, cm"
        mfc = "white"  # marker fill "white" / None
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if 'flx' in self.resAN:
            plt.plot(radii, self.resAN['flx'], 'k-', ms=MARKER_SIZE)
        if scheme.lower() == 'st':
            if 'flx' in self.resST:
                plt.errorbar(radii, self.resST['flx'], 
                             yerr=self.resST['errflx'], 
                             fmt='go', mfc=mfc, capsize=5)                
        if scheme.lower() == 'dt':
            if 'flx' in self.resST:
                plt.errorbar(radii, self.resST['flx'], 
                             yerr=self.resST['errflx'], 
                             fmt='r*', capsize=5)              
        
        plt.legend(['Analytic', scheme])
        
        plt.grid()
        plt.rc('font', size=FONT_SIZE)        # text sizes
        plt.rc('axes', labelsize=FONT_SIZE)   # labels
        plt.rc('xtick', labelsize=FONT_SIZE)  # tick labels
        plt.rc('ytick', labelsize=FONT_SIZE)  # tick labels
        

    def PlotDifferences(self):
        """plot the fluxes"""
    

        radii = self.inps['radii']
        ylabel = "Difference in Flux [%]"
        xlabel = "Distance from center, cm"
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        
        diffST = 100*(1-self.resST['flx']/self.resAN['flx'])
        diffDT = 100*(1-self.resDT['flx']/self.resAN['flx'])    
        plt.plot(radii, diffST, 'g--o', mfc='white', ms=MARKER_SIZE) 
        plt.plot(radii, diffDT, 'r--*', ms=MARKER_SIZE) 
             
        plt.legend(['ST', 'DT'])
        
        plt.grid()
        plt.rc('font', size=FONT_SIZE)        # text sizes
        plt.rc('axes', labelsize=FONT_SIZE)   # labels
        plt.rc('xtick', labelsize=FONT_SIZE)  # tick labels
        plt.rc('ytick', labelsize=FONT_SIZE)  # tick labels