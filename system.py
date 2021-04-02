import itertools
import sys

import numpy as np 
import matplotlib.pyplot as plt

import obj
import utils
import function


class Container():
    """
    Container object for simulation
    
    Parameters :
    ---------------------------------------------------
    ---------------------------------------------------
    
    objects : set of object involved in the container
    ---------------------------------------------------
    
    greens : dictionary of grenns function for two objects 
           : self.greens[obj1][obj2] returns propagation function of two objects
    ---------------------------------------------------
    
    antenna : antenna of Simulation. if other source not exists, 
              it acts as the transceiver.
    ---------------------------------------------------
    
    recorder : recorder of the field. if one wants to record the full-field, 
               it must be appended before simulation.
    ---------------------------------------------------
    
    planewavegenerator : source object. this object propagates plane wave to other object.
    
    
    Methods:
    ---------------------------------------------------
    ---------------------------------------------------
    
    append : append object to the Container. In this process, 
             Green function with other object in the container is calculated.
             
    ---------------------------------------------------
    
    propagate : calculate obj2's field by 
                (power of obj1) * (green function between obj1,obj2)
    ---------------------------------------------------
    
    experiment : Full simulation for the route of waves that propagate less than max_degree.
    
    """
    
    
    def __init__(self, k0):
        self.objects = set()
        self.greens = dict()
        self.antenna = None
        self.recorder = None
        self.planewavegenerator = None
        self.k0 = k0

    def append(self, object):
        if len(self.objects) == 0:
            assert type(object) == obj.Antenna
            
        if type(object) == obj.Antenna:
            self.antenna = object
        
        if type(object) == obj.FieldRecorder:
            self.recorder = object
            
        if type(object) == obj.PlaneWaveGenerator:
            self.planewavegenerator = object
            self.is_plane = True
        
        self.greens[object] = dict()     
        
        for other_obj in self.objects:
            if type(object) != obj.FieldRecorder:
                self.greens[object][other_obj] = object.green(self.k0, other_obj)
            if type(object) != obj.PlaneWaveGenerator:
                self.greens[other_obj][object] = other_obj.green(self.k0, object)
        self.objects.add(object)
    
    def propagate(self, obj1, obj2):
        if type(obj2) == obj.Antenna:
            E = obj1.power * self.greens[obj1][obj2]
            obj2.Ex += np.sum(E[0])
            obj2.Ey += np.sum(E[1])
            obj2.Ez += np.sum(E[2])
            
        if type(obj2) == obj.SMMarray:
            obj2.S11 = obj1.power * self.greens[obj1][obj2]

        if type(obj2) == obj.FieldRecorder:
            if (type(obj1) == obj.Antenna) or (type(obj1) == obj.PlaneWaveGenerator):
                obj2.E += obj1.power * self.greens[obj1][obj2]
            if type(obj1) == obj.SMMarray:
                obj2.E += np.sum(
                        obj1.power[np.newaxis,:,:,:,np.newaxis,np.newaxis,np.newaxis] * 
                        self.greens[obj1][obj2],
                        axis=(1,2,3))
        
    def _get_route(self, degree):
        objects_not_in_the_route = set()
        for object in self.objects:
            if object in [self.antenna, self.recorder, self.planewavegenerator]:
                objects_not_in_the_route.add(object)
        events = set(itertools.product(self.objects - objects_not_in_the_route, repeat = degree))
        event_to_remove = set()
        for event in events:
            for i in range(degree-1):
                if (event[i] == event[i+1]):
                    event_to_remove.add(event)
                    break
        events = list(events - event_to_remove)            
        return events
    
    def experiment(self, source, record = False, max_degree=4):
        if record == True:
            self.propagate(source, self.recorder)
        for degree in range(1, max_degree+1):
            events = self._get_route(degree)
            for event in events:
                for object in event:
                    self.propagate(source, object)                    
                    source = object
                self.propagate(source, self.antenna)
                if record == True:  
                    self.propagate(source, self.recorder)

