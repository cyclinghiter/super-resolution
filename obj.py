import numpy as np
import matplotlib.pyplot as plt 

import function

class _point_object:
    """ 
    Abstract point object
    
    Parameters
    ---------------------------------------------------
    ---------------------------------------------------
    x, y, z : position of point object
    
    ---------------------------------------------------
    Ex, Ey, Ez : E field of point object at position (x,y,z)
    
    
    
    Methods
    --------------------------------------------------
    --------------------------------------------------
    clear : initialize measured values
    
    ---------------------------------------------------
    displacement : calculate r vector of self and other object
    distance : absolute value of the displacement
    
    ---------------------------------------------------
    green : calculate green function using displacement and distance
    
    """
    def __init__(self, x ,y, z, mode=None):
        self.x = x
        self.y = y
        self.z = z
        self._Ex = 0
        self._Ey = 0
        self._Ez = 0 
    
    @property
    def S11(self):
        return np.sqrt(self._Ex**2 + self._Ey**2 + self._Ez**2)
    @property
    def Ex(self):
        return self._Ex
    @property
    def Ey(self):
        return self._Ey
    @property
    def Ez(self):
        return self._Ez
    
    @Ex.setter
    def Ex(self, value):
        self._Ex = value
    @Ey.setter
    def Ey(self, value):
        self._Ey = value
    @Ez.setter
    def Ez(self, value):
        self._Ez = value

    def clear(self):
        self.Ex = 0
        self.Ey = 0
        self.Ez = 0
    
    def displacement(self, object):
        return np.array([object.x - self.x,
                         object.y - self.y,
                         object.z - self.z])
        
    def distance(self, object):
        return np.sqrt((object.x - self.x)**2
                     + (object.y - self.y)**2
                     + (object.z - self.z)**2)        
    
    def green(self, k0, object, dim=3):
        r = self.distance(object)
        x = self.displacement(object)
        green_value = function.dipole_green_function(k0, r, dim=dim)
        if _point_object in type(object).mro():
            return green_value
        if type(object) == SMMarray:
            return green_value
        if type(object) == FieldRecorder:
            E = green_value * x / r[np.newaxis,:,:,:]
            return E
            
class Source(_point_object):
    
    """ Abstract class of Source object. Simply add the point object property power"""
    
    def __init__(self, x, y, z, power):
        super(Source, self).__init__(x ,y, z)
        self._power = power

    @property
    def power(self):
        return self._power
    
    @power.setter
    def power(self, value):
        self._power = value

class Antenna(Source):
    """ Antenna Object. same as Source."""
    
    def __init__(self, x, y, z , power):
        super(Antenna, self).__init__(x,y,z, power)

    def copy(self):
        new = Antenna(self.x, self.y, self.z, self.power)
        return new

class Dipole(Source):
    """passive source. act like antenna if E field in. else power equals 0. """
    def __init__(self, x, y, z, alpha):
        super(Dipole, self).__init__(x,y,z, 0)
        self.alpha = alpha
        
    @property
    def Ex(self):
        return self._Ex
    @property
    def Ey(self):
        return self._Ey
    @property
    def Ez(self):
        return self._Ez
    
    @Ex.setter
    def Ex(self, value):
        self._Ex = value 
        self.power = self.alpha * self.S11

    @Ey.setter
    def Ey(self, value):
        self._Ey = value 
        self.power = self.alpha * self.S11
        
    @Ez.setter
    def Ez(self, value):
        self._Ez = value 
        self.power = self.alpha * self.S11        

    def clear(self):
        self.Ex = 0
        self.Ey = 0
        self.Ez = 0 

class _array_object():
    """
    Abstract array object. contains SMM and FieldRecorder.
    Main function of this class is calculating green function with other objects.
    
    
    Parameters
    ---------------------------------------------------
    ---------------------------------------------------
    x, y, z : position array. dtype : ndarray, ndim = 3
        
    Methods
    --------------------------------------------------
    --------------------------------------------------
    clear : initialize measured values
    
    ---------------------------------------------------
    displacement : calculate r vector of self and other object
    distance : absolute value of the displacement
    
    ---------------------------------------------------
    green : calculate green function using displacement and distance
    
    """
    
    def __init__(self, x, y, z):
        self.x, self.y, self.z = np.meshgrid(x,y,z)
        self.shape = self.x.shape
    
    def displacement(self, object):
        return np.array([object.x - self.x,
                         object.y - self.y,
                         object.z - self.z])
        
    def distance(self, object):
        return np.sqrt((object.x - self.x)**2 \
                     + (object.y - self.y)**2 \
                     + (object.z - self.z)**2)        
    
    def green(self, k0, object, dim=3):
        if _point_object in type(object).mro():
            r = self.distance(object)
            x = self.displacement(object)
            green_value = function.dipole_green_function(k0, r, dim=dim)
            E = green_value * x / r[np.newaxis,:,:,:]
            
        if type(object) == FieldRecorder:
            x = object.x[np.newaxis,np.newaxis,np.newaxis,:,:,:] - self.x[:,:,:,np.newaxis,np.newaxis, np.newaxis]
            y = object.y[np.newaxis,np.newaxis,np.newaxis,:,:,:] - self.y[:,:,:,np.newaxis,np.newaxis, np.newaxis]
            z = object.z[np.newaxis,np.newaxis,np.newaxis,:,:,:] - self.z[:,:,:,np.newaxis,np.newaxis, np.newaxis]
            r = np.sqrt(x**2 + y**2 + z**2)
            r_vec = np.array([x,y,z])
            green_value = function.dipole_green_function(k0, r, dim=dim)
            E = green_value * r_vec / r[np.newaxis,:,:,:]
        return E    
        
class SMMarray(_array_object):
    """
    SMM cell array. it's normal vector is always on the z axis.
    it's green function is approximated as Dipole. 
    Modulated E field is determined only the strength of the E field. 
    It is invariant with the direciton of EM wave.
    
    the cell which is switched on does not change phase of a wave, but the amplitude is reduced
    the cell which is switched off does not change amplitude of a wave, but changes the phase. 
    
    Parameters
    ---------------------------------------------------
    ---------------------------------------------------
    x, y, z : center of position
    ---------------------------------------------------
    dx, dy: distance between two cell. 
    ---------------------------------------------------
    k0 : default modulation wavevector.
    ---------------------------------------------------
    pattern : pattern of SMMarray
    ---------------------------------------------------
    amp : power amplitude by modulation if cell is on
    ---------------------------------------------------
    phase : phase shift by modulation if cell is off 
    
    
    Method:
    ---------------------------------------------------
    ---------------------------------------------------
    Green function calculation is the main function of this object. 
    it is defined in its parent class, _array_object
    
    
    """
    def __init__(self, x, y ,z, dx, dy, k0, pattern, amp = 0.7, phase = np.pi):
        self.dx = dx
        self.dy = dy
        self.k0 = k0
        self.amp = amp
        self.z = z
        self.pattern = pattern.reshape(*pattern.shape, 1)
        self.phase = phase
        self.x_array = np.arange(-pattern.shape[0]//2+0.5, pattern.shape[0]//2, 1) * dx + x
        self.y_array = np.arange(-pattern.shape[1]//2+0.5, pattern.shape[1]//2, 1) * dy + y
        self.x, self.y, self.z = np.meshgrid(self.y_array, self.x_array, self.z)
        self.shape = self.x.shape
        
        self._S11 = np.zeros(self.shape, dtype='complex128')
        self.power = np.zeros(self.shape, dtype = 'complex128')
        
    @property
    def S11(self):
        return self._S11
    
    @S11.setter
    def S11(self, value):
        self._S11 = value
        self.power = self.amp * self.S11 * self.pattern + \
                     self.S11 * (1-self.pattern) * np.exp(1J*self.k0*self.phase) 

    def clear(self):
        self.S11 = np.zeros(self.shape, dtype='complex128')

class FieldRecorder(_array_object):
    def __init__(self, x, y, z):
        super(FieldRecorder, self).__init__(x,y,z)
        self._E = np.zeros((3,*self.shape), dtype='complex128')

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, value):
        self._E = value

    def clear(self):
        self.E = np.zeros((3,*self.shape), dtype='complex128')
    # def clear(self):
    #     self.S11 = np.zeros(self.shape, dtype='complex128')
        
class PlaneWaveGenerator(Source):
    """
    plane wave generator which can determine wave vector.
    Distance between the plane that has normal vector k and position (x,y,z) and a point
    determines phase of the wave.
    
    """
    def __init__(self, x, y, z, kx, ky, kz, power=1):
        super(PlaneWaveGenerator, self).__init__(self, x,y,z, power)
        self.kx = kx
        self.ky = ky
        self.kz = kz
        
    @property
    def k0(self):
        return np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
    
    @property
    def n_vec(self):
        return np.array([self.kx, self.ky, self.kz]) / self.k0
            
    def distance(self, object):
        x = self.n_vec[0] * object.x - self.x
        y = self.n_vec[1] * object.y - self.y
        z = self.n_vec[2] * object.z - self.z
        distance = np.abs(x + y + z)
        return distance
    
    def green(self, k0, object):
        r = self.distance(object)
        green = np.exp(1J * self.k0 * r)
        return green
        