#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import sys
import warnings
import io


import numpy as np

import pandas as pd
# import pyreadr
import time
import datetime

import math
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set_style('darkgrid')


import gc
import json 

# !pip install pathos
from pathos.multiprocessing import ProcessPool
from multiprocessing import Lock

import multiprocessing
# from multiprocessing import Lock,Pool

import pickle

import random
random.seed(42)
np.random.seed(42)


import string 
import copy
#!pip install pygmo
from pygmo import hypervolume as pyhv   

from tqdm.notebook import tqdm

# data aug
import tsaug
from tsaug.visualization import plot
from tsaug import AddNoise, TimeWarp, Convolve, Drift, Pool, Crop

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix

from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# --------------
# !pip install treelib
from treelib import Node, Tree
from sklearn.tree import _tree
# -------------------

# suppress stdout and stderr
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

## import local lib jaccard_test
module_path = os.path.abspath(os.path.join('./jaccard/jaccard_test'))
if module_path not in sys.path:
    sys.path.append(module_path)

from jaccard import jaccard_test

from operator import attrgetter
import operator

import re

# import deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from inspect import isclass
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt

from functools import partial

from tqdm import tqdm

# rtamt repos version 
# https://github.com/nickovic/rtamt

import rtamt # dense/discrete time online and offline library
from rtamt.pastifier.stl.horizon import StlHorizon
from rtamt import RTAMTException

max_trace_length = 100000 # not used: 100 for debug


def load_pickle(file):
    try:
        pickle_in= open(file, "rb")
        data = pickle.load(pickle_in)
        pickle_in.close()
        return data
    except IOError:
        print("Error loading pickle data from " + file)
    return None

def write_pickle(data, file):
    try:
        pickle_out = open(file, "wb")
        pickle.dump(data, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        return True
    except IOError:
        print("Error writing data to " + file)
    return False


def dominates(a, b):
    dominates = True
    different = False
    for i in range(len(a)):
        if not(a[i] >= b[i]):
            dominates = False
        if a[i] != b[i]:
            different = True
    return dominates and different        

def parallel_extract_all_front(element_fitness): # reference fitness value and all fitnesses
    all_fitnesses = GP_STL.temp
    is_dominated = False
    for yfit in all_fitnesses:
        if dominates(yfit, element_fitness):
            is_dominated = True
            break
    return is_dominated
   

def eval_wrapper(args):# par, traces_data, fitness_map
    return gp_instance.toolbox.evaluate(*args) 





class GP_STL():
    
    """ 
        Genetic Programming Algorithm class for the extraction of STL properties
        This implementation is based on deap a python library for genetic programming. 
    """
        
    min_height = 1
    max_height = None
    max_gen_height = 6
        
    epsilon=1e-3

    temp = None
    reverse_obj_window = None

    signal_columns = None

    n_train_good_traces = None

    column_types = None

    good_witnesses_ea = None
    bad_witnesses_ea = None    

    hv_calc_good_witnesses_ea = None
    hv_calc_bad_witnesses_ea = None

    goodbad_traces_at_ea_generation = 0
    num_good_witnesses = None
    num_bad_witnesses = None

    margin_weight = None

    ea_score_method = None

    maxfar = None

    ea_patience = None
    mutation_decay = None
    mutate_all_or_one = None
    
    accepted_types = ['float']#['int', 'float']
    
    def __init__(self, data, signal_columns, processes, max_horizon, eval_wrapper, train_good_traces=[], seed_in=42, exploit_idempotence=True,
                cross_prob_terminal_node = 0.1, max_height= 17, margin_weight = 1.0, goodbad_traces_at_ea_generation = 0, ea_score_method="FAR", 
                 ea_patience=0, mutation_decay=2, mutate_all_or_one='all'):
        
        """ 
            Genetic Programming Algorithm definition and initalization
            
            Required parameters:
        
        
                data - the considered dataset
                signal_columns - columns considered for the property generation process
            
        """

        random.seed(seed_in)
        np.random.seed(seed_in)
        
        #self.data = data

        GP_STL.ea_patience = ea_patience
        GP_STL.mutation_decay = mutation_decay
        GP_STL.mutate_all_or_one = mutate_all_or_one

        GP_STL.max_height = max_height

        GP_STL.signal_columns = signal_columns
        
        GP_STL.processes = processes
        
        self.max_horizon = max_horizon

        GP_STL.margin_weight = margin_weight
        
        self.eval_wrapper = eval_wrapper

        GP_STL.goodbad_traces_at_ea_generation = goodbad_traces_at_ea_generation
        
        #self.max_range = max(np.max(data[signal_columns], axis=0) - np.min(data[signal_columns], axis=0))
        
        # debug only  
        self.train_good_traces = train_good_traces
        GP_STL.n_train_good_traces = len(self.train_good_traces) 

        GP_STL.ea_score_method = ea_score_method

        self.exploit_idempotence = exploit_idempotence
        self.cross_prob_terminal_node = cross_prob_terminal_node
        
        GP_STL.column_types = {}
               
        for c in GP_STL.signal_columns:
            col_type = re.sub(r'[0-9]+', '', data[c].dtype.name)
            if not col_type in  GP_STL.accepted_types:
                raise TypeError("Column '"+c+"' type not supported yet:" + col_type)
            GP_STL.column_types[c] = col_type


        if self.exploit_idempotence:
            # Generic (sub)-Formula Type definition 
            globals()["FormulaType"] = type("FormulaType", (gp.Primitive,),{})
            globals()["FormulaType.ret"] = globals()["FormulaType"]
    
            globals()["FormulaTypeSurelyNoF"] = type("FormulaTypeSurelyNoF", (globals()["FormulaType"],),{})
            globals()["FormulaTypeSurelyNoF.ret"] = globals()["FormulaTypeSurelyNoF"]
            
            globals()["FormulaTypeSurelyNoG"] = type("FormulaTypeSurelyNoG", (globals()["FormulaType"],),{})
            globals()["FormulaTypeSurelyNoG.ret"] = globals()["FormulaTypeSurelyNoG"] 
    
            globals()["FormulaTypeSurelyNoNot"] = type("FormulaTypeSurelyNoNot", (globals()["FormulaType"],),{})
            globals()["FormulaTypeSurelyNoNot.ret"] = globals()["FormulaTypeSurelyNoNot"] 
            
            globals()["FormulaTypeSurelyNoFG"] = type("FormulaTypeSurelyNoFG", (globals()["FormulaTypeSurelyNoF"],globals()["FormulaTypeSurelyNoG"],),{})
            globals()["FormulaTypeSurelyNoFG.ret"] = globals()["FormulaTypeSurelyNoFG"] 
    
            globals()["FormulaTypeSurelyNoFNot"] = type("FormulaTypeSurelyNoFNot", (globals()["FormulaTypeSurelyNoF"],globals()["FormulaTypeSurelyNoNot"],),{})
            globals()["FormulaTypeSurelyNoFNot.ret"] = globals()["FormulaTypeSurelyNoFNot"] 
    
            globals()["FormulaTypeSurelyNoGNot"] = type("FormulaTypeSurelyNoGNot", (globals()["FormulaTypeSurelyNoG"],globals()["FormulaTypeSurelyNoNot"],),{})
            globals()["FormulaTypeSurelyNoGNot.ret"] = globals()["FormulaTypeSurelyNoGNot"] 
    
            globals()["FormulaTypeSurelyNoFGNot"] = type("FormulaTypeSurelyNoFGNot", (globals()["FormulaTypeSurelyNoF"],globals()["FormulaTypeSurelyNoG"],globals()["FormulaTypeSurelyNoNot"],),{})
            globals()["FormulaTypeSurelyNoFGNot.ret"] = globals()["FormulaTypeSurelyNoFGNot"] 
    
    
            globals()["ConstraintOP"] = type("ConstraintOP", (globals()["FormulaType"],),{})
            globals()["ConstraintOP.ret"] = globals()["ConstraintOP"]
    
            # pset = gp.PrimitiveSetTyped("formula", [type(data[c].iloc[0]) for c in sensor_columns], FormulaType)
            self.pset = gp.PrimitiveSetTyped("formula", [], globals()["FormulaType"])
            
    
            # STL Operators Interval Bounds definition 
            globals()['ModalLBoundEphemeral'] = type("ModalLBoundEphemeral", (gp.Terminal,), {})
            globals()['ModalLBoundEphemeral'].ret = globals()['ModalLBoundEphemeral']
            globals()['_ModalLBoundEphemeral'] = (lambda : random.random())  
            globals()['_ModalLBoundEphemeral'].__name__ = '_ModalLBoundEphemeral'
    
            globals()['ModalRBoundEphemeral'] = type("ModalRBoundEphemeral", (gp.Terminal,), {})
            globals()['ModalRBoundEphemeral'].ret = globals()['ModalRBoundEphemeral']
            globals()['_ModalRBoundEphemeral'] = (lambda : random.random())
            globals()['_ModalRBoundEphemeral'].__name__ = '_ModalRBoundEphemeral'
    
            self.pset.addEphemeralConstant("ModalLBoundEphemeral", globals()["_ModalLBoundEphemeral"], globals()["ModalLBoundEphemeral"])
            self.pset.addPrimitive(globals()["_ModalLBoundEphemeral"], [], globals()["ModalLBoundEphemeral"])
            self.pset.addEphemeralConstant("ModalRBoundEphemeral", globals()["_ModalRBoundEphemeral"], globals()["ModalRBoundEphemeral"])
            self.pset.addPrimitive(globals()["_ModalRBoundEphemeral"], [], globals()["ModalRBoundEphemeral"])
    
            # STL Primitives definition
            def F_op(a: globals()["ModalLBoundEphemeral"], b: globals()["ModalRBoundEphemeral"], c: globals()["FormulaTypeSurelyNoF"]) -> globals()["FormulaTypeSurelyNoGNot"]: pass 
            def G_op(a: globals()["ModalLBoundEphemeral"], b: globals()["ModalRBoundEphemeral"], c: globals()["FormulaTypeSurelyNoG"]) -> globals()["FormulaTypeSurelyNoFNot"]: pass     
            def U_op(a: globals()["ModalLBoundEphemeral"], b: globals()["ModalRBoundEphemeral"], c: globals()["FormulaType"], d: globals()["FormulaType"]) -> globals()["FormulaTypeSurelyNoFGNot"]: pass
            def not_op(a: globals()["FormulaTypeSurelyNoNot"]) -> globals()["FormulaTypeSurelyNoFG"]: pass
            def or_op(a: globals()["FormulaType"], b: globals()["FormulaType"]) -> globals()["FormulaTypeSurelyNoFGNot"]: pass
            def and_op(a: globals()["FormulaType"], b: globals()["FormulaType"]) -> globals()["FormulaTypeSurelyNoFGNot"]: pass
            def imp_op(a: globals()["FormulaType"], b: globals()["FormulaType"]) -> globals()["FormulaTypeSurelyNoFGNot"]: pass
    
    
            self.pset.addPrimitive(F_op, [globals()["ModalLBoundEphemeral"], globals()["ModalRBoundEphemeral"], globals()["FormulaTypeSurelyNoF"]], globals()["FormulaTypeSurelyNoGNot"])
            self.pset.addPrimitive(G_op, [globals()["ModalLBoundEphemeral"], globals()["ModalRBoundEphemeral"], globals()["FormulaTypeSurelyNoG"]], globals()["FormulaTypeSurelyNoFNot"])
            self.pset.addPrimitive(U_op, [globals()["ModalLBoundEphemeral"], globals()["ModalRBoundEphemeral"], globals()["FormulaType"], globals()["FormulaType"]], globals()["FormulaTypeSurelyNoFGNot"])
            self.pset.addPrimitive(not_op, [globals()["FormulaTypeSurelyNoNot"]], globals()["FormulaTypeSurelyNoFG"])
            self.pset.addPrimitive(or_op, [globals()["FormulaType"], globals()["FormulaType"]], globals()["FormulaTypeSurelyNoFGNot"])
            self.pset.addPrimitive(and_op, [globals()["FormulaType"], globals()["FormulaType"]], globals()["FormulaTypeSurelyNoFGNot"])
            self.pset.addPrimitive(imp_op, [globals()["FormulaType"], globals()["FormulaType"]], globals()["FormulaTypeSurelyNoFGNot"])

        else:
            
            # Generic (sub)-Formula Type definition 
            globals()["FormulaType"] = type("FormulaType", (gp.Primitive,),{})
            globals()["FormulaType.ret"] = globals()["FormulaType"]
    
            globals()["ConstraintOP"] = type("ConstraintOP", (globals()["FormulaType"],),{})
            globals()["ConstraintOP.ret"] = globals()["ConstraintOP"]
    
            # pset = gp.PrimitiveSetTyped("formula", [type(data[c].iloc[0]) for c in sensor_columns], FormulaType)
            self.pset = gp.PrimitiveSetTyped("formula", [], globals()["FormulaType"])
            
    
            # STL Operators Interval Bounds definition 
            globals()['ModalLBoundEphemeral'] = type("ModalLBoundEphemeral", (gp.Terminal,), {})
            globals()['ModalLBoundEphemeral'].ret = globals()['ModalLBoundEphemeral']
            globals()['_ModalLBoundEphemeral'] = (lambda : random.random())  
            globals()['_ModalLBoundEphemeral'].__name__ = '_ModalLBoundEphemeral'
    
            globals()['ModalRBoundEphemeral'] = type("ModalRBoundEphemeral", (gp.Terminal,), {})
            globals()['ModalRBoundEphemeral'].ret = globals()['ModalRBoundEphemeral']
            globals()['_ModalRBoundEphemeral'] = (lambda : random.random())
            globals()['_ModalRBoundEphemeral'].__name__ = '_ModalRBoundEphemeral'
    
            self.pset.addEphemeralConstant("ModalLBoundEphemeral", globals()["_ModalLBoundEphemeral"], globals()["ModalLBoundEphemeral"])
            self.pset.addPrimitive(globals()["_ModalLBoundEphemeral"], [], globals()["ModalLBoundEphemeral"])
            self.pset.addEphemeralConstant("ModalRBoundEphemeral", globals()["_ModalRBoundEphemeral"], globals()["ModalRBoundEphemeral"])
            self.pset.addPrimitive(globals()["_ModalRBoundEphemeral"], [], globals()["ModalRBoundEphemeral"])
    
            # STL Primitives definition
            def F_op(a: globals()["ModalLBoundEphemeral"], b: globals()["ModalRBoundEphemeral"], c: globals()["FormulaType"]) -> globals()["FormulaType"]: pass 
            def G_op(a: globals()["ModalLBoundEphemeral"], b: globals()["ModalRBoundEphemeral"], c: globals()["FormulaType"]) -> globals()["FormulaType"]: pass     
            def U_op(a: globals()["ModalLBoundEphemeral"], b: globals()["ModalRBoundEphemeral"], c: globals()["FormulaType"], d: globals()["FormulaType"]) -> globals()["FormulaType"]: pass
            def not_op(a: globals()["FormulaType"]) -> globals()["FormulaType"]: pass
            def or_op(a: globals()["FormulaType"], b: globals()["FormulaType"]) -> globals()["FormulaType"]: pass
            def and_op(a: globals()["FormulaType"], b: globals()["FormulaType"]) -> globals()["FormulaType"]: pass
            def imp_op(a: globals()["FormulaType"], b: globals()["FormulaType"]) -> globals()["FormulaType"]: pass
            self.pset.addPrimitive(F_op, [globals()["ModalLBoundEphemeral"], globals()["ModalRBoundEphemeral"], globals()["FormulaType"]], globals()["FormulaType"])
            self.pset.addPrimitive(G_op, [globals()["ModalLBoundEphemeral"], globals()["ModalRBoundEphemeral"], globals()["FormulaType"]], globals()["FormulaType"])
            self.pset.addPrimitive(U_op, [globals()["ModalLBoundEphemeral"], globals()["ModalRBoundEphemeral"], globals()["FormulaType"], globals()["FormulaType"]], globals()["FormulaType"])
            self.pset.addPrimitive(not_op, [globals()["FormulaType"]], globals()["FormulaType"])
            self.pset.addPrimitive(or_op, [globals()["FormulaType"], globals()["FormulaType"]], globals()["FormulaType"])
            self.pset.addPrimitive(and_op, [globals()["FormulaType"], globals()["FormulaType"]], globals()["FormulaType"])
            self.pset.addPrimitive(imp_op, [globals()["FormulaType"], globals()["FormulaType"]], globals()["FormulaType"])            

        
        # STL Constraint Primitives definition

        for i, c in enumerate(GP_STL.signal_columns):
            globals()[c+"ConstraintLVal"] = type(c+"ConstraintLVal", (gp.Terminal,), {})
            globals()[c+"ConstraintLVal"].ret = globals()[c+"ConstraintLVal"]
            globals()["_"+c+"ConstraintLVal"] = (lambda: c)
            globals()["_"+c+"ConstraintLVal"].__name__ = "_"+c+"ConstraintLVal"
            self.pset.addTerminal(globals()[c+"ConstraintLVal"](c, False, globals()[c+"ConstraintLVal"]), globals()[c+"ConstraintLVal"], name=c)
            self.pset.addPrimitive(globals()["_"+c+"ConstraintLVal"], [], globals()[c+"ConstraintLVal"])
            
            
            globals()[c+"Ephemeral"] = type(c+"Ephemeral", (gp.Ephemeral,), {})
            globals()[c+"Ephemeral"].ret = globals()[c+"Ephemeral"]
            # globals()["_"+c+"Ephemeral"] = (lambda: random.uniform(self.data[c].min(), self.data[c].max()))
            globals()["_"+c+"Ephemeral"] = (lambda: random.uniform(0, 1))
            globals()["_"+c+"Ephemeral"].__name__ = "_"+c+"Ephemeral"
            self.pset.addEphemeralConstant(c+"Ephemeral", globals()["_"+c+"Ephemeral"], globals()[c+"Ephemeral"])
            self.pset.addPrimitive(globals()["_"+c+"Ephemeral"], [], globals()[c+"Ephemeral"])

            globals()[c+"_ge_op"] = (lambda a, b : None)
            globals()[c+"_ge_op"].__name__ = c+"_ge_op"
            
            self.pset.addPrimitive(globals()[c+"_ge_op"], [globals()[c+"ConstraintLVal"], globals()[c+"Ephemeral"]], globals()["ConstraintOP"])
        

        # Now create the Individual class
        creator.create("fitness", base.Fitness, weights=(+1.0,+1.0,+1.0)) # minimize a single objective (-1) or maximize it (+1)
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.fitness, pset=self.pset)
        self.toolbox = base.Toolbox() # this is a c for all individuals, functions, operators, etc.

        
        for i in range(GP_STL.min_height, GP_STL.max_gen_height+1):
            self.toolbox.register("expr"+str(i), GP_STL.gen_safe, pset=self.pset, min_= i, max_ = i) # gp.genHalfAndHalf generator function to initialize the _ulation individuals. "genHalfAndHalf" generates the trees in a list format, parameters: min_height, max_height
            self.toolbox.register("individual"+str(i), tools.initIterate, creator.Individual, getattr(self.toolbox, "expr"+str(i))) # initializer for the elements of the population: a weight for each fingerprint distance function
            self.toolbox.register("population"+str(i), tools.initRepeat, list, getattr(self.toolbox, "individual"+str(i))) # initializer for the population        

        
        ############################
        
        
        # Now define the genetic operators
        self.toolbox.register("evaluate", self.evalIndividual)

        self.toolbox.register("select", tools.selNSGA2) 
        self.toolbox.register("mate", self.random_crossover_operator) # in-place operation
        self.toolbox.register('mutate', self.random_mutation_operator, pset=self.pset)


    ### end __init__


    ########################## static methods 
    
    # @staticmethod
    # def get_split_window(relative_window, trace_length):
    #     return int((trace_length - 2)*relative_window)+1

    @staticmethod
    def eval_worker_wrapper(args):
        # Unpack 'self' and 'arg' and call the instance method
        instance, arg = args
        return instance.toolbox.evaluate(*arg)
   
    
    #@staticmethod
    def random_mutation_operator(self, individual, roll, pset):
        '''
            Randomly picks a replacement, or ephemeral mutation.
        '''
        if roll <= 1./4.:
            mut = gp.mutNodeReplacement(individual, pset=pset)
        elif roll <= 2./4.:
            if round((roll*1000)) % 2 == 0:
                individual.window = self.gen_split_window()            
                mut = [individual]
            else:
                individual.window = max(1, min(individual.window+int(np.random.normal(loc=0.0, scale=10)), self.t_len-1))
                mut = [individual]
        elif roll <= 3./4.:
            mut = gp.mutShrink(individual)
        else:
            mut = gp.mutEphemeral(individual, GP_STL.mutate_all_or_one)

        return mut


    def random_crossover_operator(self, ind1, ind2, roll):
        '''
            Randomly picks a replacement, or ephemeral mutation.
        '''
        if roll <= 0.5:
            tmp_ind1 = copy.deepcopy(ind1)
            tmp_ind2 = copy.deepcopy(ind2)
            child1, child2 = GP_STL.cxOnePointLeafBiased(ind1, ind2, self.cross_prob_terminal_node) # as said by Koza <--- FAR 0 con 0.9 ????
            if child1.height > GP_STL.max_height:
                child1 = tmp_ind1
            if child2.height > GP_STL.max_height:
                child2 = tmp_ind2
        else:
            child1 = copy.deepcopy(ind1)
            child2 = copy.deepcopy(ind2)   
            help_w = child1.window
            child1.window = child2.window
            child2.window = help_w
        return child1, child2
        
    
    @staticmethod
    def genFull(pset, min_, max_, type_=None):
        """Generate an expression where each leaf has the same depth
        between *min* and *max*.
        :param pset: Primitive set from which primitives are selected.
        :param min_: Minimum height of the produced trees.
        :param max_: Maximum Height of the produced trees.
        :param type_: The type that should return the tree when called, when
                      :obj:`None` (default) the type of :pset: (pset.ret)
                      is assumed.
        :returns: A full tree with all leaves at the same depth.
        """

        def condition(height, depth):
            """Expression generation stops when the depth is equal to height."""
            return depth == height

        return GP_STL.generate(pset, min_, max_, condition, type_)

    
    @staticmethod
    def generate(pset, min_, max_, condition, type_=None):
        """Generate a tree as a list of primitives and terminals in a depth-first
        order. The tree is built from the root to the leaves, and it stops growing
        the current branch when the *condition* is fulfilled: in which case, it
        back-tracks, then tries to grow another branch until the *condition* is
        fulfilled again, and so on. The returned list can then be passed to the
        constructor of the class *PrimitiveTree* to build an actual tree object.
        :param pset: Primitive set from which primitives are selected.
        :param min_: Minimum height of the produced trees.
        :param max_: Maximum Height of the produced trees.
        :param condition: The condition is a function that takes two arguments,
                          the height of the tree to build and the current
                          depth in the tree.
        :param type_: The type that should return the tree when called, when
                      :obj:`None` (default) the type of :pset: (pset.ret)
                      is assumed.
        :returns: A grown tree with leaves at possibly different depths
                  depending on the condition function.
        """
        if type_ is None:
            type_ = pset.ret
        expr = []
        height = random.randint(min_, max_)
        stack = [(0, type_)]
        
        while len(stack) != 0:
            depth, type_ = stack.pop()
            if condition(height, depth):
                try:
                    term = random.choice(pset.terminals[type_])                 
                    if isclass(term):
                        term = term()
                    expr.append(term)
                except IndexError:
                    error = True
                    prims = [pr for pr in pset.primitives[type_] if pr.arity == 0]
                    if len(prims) > 0:
                        prim = random.choice(prims)
                        expr.append(prim)                    
                    else:
                        _, _, traceback = sys.exc_info()
#                         print('pset.terminals for type "', type_, '":',  pset.terminals[type_])
                        raise IndexError("The gp.generate function tried to add "
                                          "a terminal of type '%s', but there is "
                                          "none available." % (type_,)).with_traceback(traceback)                    
            else:                
                try:
                    if height - depth <= 1 and type_.__name__ == "FormulaType" :
                        prim = random.choice(pset.primitives[globals()["ConstraintOP"]])
                    else:
                        if random.random() > 1/len(pset.primitives[type_]):
                            prim = random.choice([pr for pr in pset.primitives[type_] if ( pr.arity > 0 and pr.ret.__name__ != "ConstraintOP" ) ])
                        else:
                            prim = random.choice([pr for pr in pset.primitives[type_] if pr.arity > 0 ])
                    expr.append(prim)
                    for arg in reversed(prim.args):
                        stack.append((depth + 1, arg))
                except IndexError:                    
                    terms = pset.terminals[type_]
                    if len(terms) > 0:
                        term = random.choice(terms)
                        if isclass(term):
                            term = term()
                        expr.append(term)
                    else:
                        _, _, traceback = sys.exc_info()
#                         print('pset.primitives:', pset.primitives[type_])
                        raise IndexError("The gp.generate function tried to add "
                                          "a primitive of type '%s', but there is "
                                          "none available." % (type_,)).with_traceback(traceback)
               
        return expr

    
    
    @staticmethod
    def get_horizon_from_mon(mon):
        h = StlHorizon()
        for s in mon.ast.specs:
            horizon = h.visit(s, None)
        #return math.ceil(horizon) #if horizon > 1. else 1
        assert float(horizon).is_integer()
        return int(horizon)

    @staticmethod
    def get_tree_height(t):
        """Return the height of the tree, or the depth of the 
           deepest node.
        """
        stack = [0]
        max_depth = 0
        for elem in t:
            depth = stack.pop()
            max_depth = max(max_depth, depth)

            if elem.arity > 0: 
                stack.extend([depth + 1] * elem.arity)
        return max_depth
    
    @staticmethod
    def gen_safe(pset, min_, max_):
        
        """
            Generate a feasible individual given a Primitive set, a min height and a max height.
            
            :param pset: instance of gp.PrimitiveSetTyped representing the strongly typed primitive set
            :param min_: min height of the generated individual
            :param max_: max height of the generated individual
                
            :return: generated individual 
                
        """
        
        g=True
        while g:
            g = False
            try:
                expr = GP_STL.genFull(pset, min_=min_, max_=max_)

                expr = GP_STL.replace_ephemerals(expr, pset)
                if not GP_STL.feasible(expr):
                    raise ValueError("Not feasible individual: Lbound > Rbound")
                
                h=GP_STL.get_tree_height(expr)
                if h < min_ or h > max_:
                    raise ValueError("Height constraints violated")

            except Exception as e:
                g = True
        return expr



    @staticmethod
    def feasible(individual):
        
        """
            For a given individual checks that ModalLBound <= ModalRBound.
            
            :param individual: instance of gp.PrimitiveTree representing a population individual
            
            :return: True if the inidividual is feasible, False otherwise
        
        """

        last_LBound = -1

        for e in individual:
            if e.ret == globals()['ModalLBoundEphemeral']:
                last_LBound = e.value

            if e.ret == globals()['ModalRBoundEphemeral']:
                if e.value >= last_LBound:
                    last_LBound = -1
                else:
                    return False
        return True

    @staticmethod
    def replace_ephemerals(formula_tree, pset):
        """
            Given a formula tree individual replaces all the Ephemeral an Terminal 
            Primirtives (added in generation phase to overcame the well known deap 
            Terminal generation bug, see https://github.com/DEAP/deap/issues/237 
            for more details). This function MUST be called BEFORE evaluating 
            the fitness of an individual.
            
            :param formula_tree: the individual to process
            :return: individual with dummy terminals related primitives replaced 
                    by real Terminal instances
        
        """    
        
        for i,e in enumerate(formula_tree):
            if isinstance(e, sys.modules['deap'].gp.Primitive) and str.startswith(e.name, '_'):
                if str.endswith(e.name,"Ephemeral"):
                    formula_tree[i] =  pset.terminals[globals()[e.name[1:]]][0]()
                else: # Terminal
                    formula_tree[i] =  pset.terminals[globals()[e.name[1:]]][0]
        return  formula_tree

    @staticmethod
    def translate_tree(s, signal_columns):
        s=s.replace('(','[')
        s=s.replace(')',']')

        s=re.sub(r'_op[0-9]*', '', s)
        
        for c in signal_columns:
            s=s.replace(c+'_', '')
            s=s.replace(c, "'"+c+"'")

        s = re.sub(r"([a-zA-Z0-9_]+)\[", r"['\1', ", s)
        # s = re.sub(r"([a-zA-Z0-9_\-\.]+)([,\]])", r"'\1'\2", s) 
        ls= eval(s)
        return ls
    
    @staticmethod
    def get_stl_expr(formula_tree, window, scaler=None): ### if scaler != None => denormalize formula
        
        if type(formula_tree) == list  and  len(formula_tree) > 0: 
        
            if formula_tree[0] == 'F':
                if len(formula_tree) != 4:
                    raise ValueError("Syntax error" + str(formula_tree))
                return "eventually["+ str(int(formula_tree[1]*window)) + ":" + str(int(formula_tree[2]*window)) + "] (" + GP_STL.get_stl_expr(formula_tree[3], window, scaler) + ")"
            
            if formula_tree[0] == 'G':
                if len(formula_tree) != 4:
                    raise ValueError("Syntax error" + str(formula_tree))
                return "always["+ str(int(formula_tree[1]*window)) + ":" + str(int(formula_tree[2]*window)) + "] (" + GP_STL.get_stl_expr(formula_tree[3], window, scaler) + ")"

            
            if formula_tree[0] == 'U':
                if len(formula_tree) != 5:
                    raise ValueError("Syntax error" + str(formula_tree))
                return "(" + GP_STL.get_stl_expr(formula_tree[3], window, scaler) + ") until["+ str(int(formula_tree[1]*window)) + ":" + str(int(formula_tree[2]*window)) + "] (" + GP_STL.get_stl_expr(formula_tree[4], window, scaler) + ")"
                
            if formula_tree[0] == 'not':
                if len(formula_tree) != 2:
                    raise ValueError("Syntax error" + str(formula_tree))
                return "not (" + GP_STL.get_stl_expr(formula_tree[1], window, scaler) + ")"
            
            if formula_tree[0] == 'or':
                if len(formula_tree) != 3:
                    raise ValueError("Syntax error" + str(formula_tree))
                return "(" + GP_STL.get_stl_expr(formula_tree[1], window, scaler) + ") or (" + GP_STL.get_stl_expr(formula_tree[2], window, scaler) + ")"
            
            if formula_tree[0] == 'and':
                if len(formula_tree) != 3:
                    raise ValueError("Syntax error" + str(formula_tree))
                return "(" + GP_STL.get_stl_expr(formula_tree[1], window, scaler) + ") and (" + GP_STL.get_stl_expr(formula_tree[2], window, scaler) + ")"
            
            if formula_tree[0] == 'imp':
                if len(formula_tree) != 3:
                    raise ValueError("Syntax error" + str(formula_tree))
                return "(" + GP_STL.get_stl_expr(formula_tree[1], window, scaler) + ") -> (" + GP_STL.get_stl_expr(formula_tree[2], window, scaler) + ")"
        
            if formula_tree[0] == 'ge': # base case ">="
                if len(formula_tree) != 3:
                    raise ValueError("Syntax error" + str(formula_tree))
                
                val = formula_tree[2]
                if(scaler!=None):
                    val = GP_STL.denormalize_constant(scaler, formula_tree[1], formula_tree[2]) # N.B.: [1:-1] remove heading and ending "'"
         
                return str(formula_tree[1]) + " >= " + str(val)


            else:
                raise ValueError("Syntax error" + str(formula_tree))
        else:
            raise ValueError("Syntax error" + str(formula_tree))


    @staticmethod
    def normalize_traces_data(traces_data, columns, train_good_traces, good_witnesses_ea, bad_witnesses_ea):
        # Set up the normalization
        scl = MinMaxScaler()
        for trace in traces_data:
            scl.partial_fit(trace[columns])

        ##### 
        for trace in good_witnesses_ea:
            scl.partial_fit(trace[columns])
        #####
        for trace in bad_witnesses_ea:
            scl.partial_fit(trace[columns])
        #####

        for trace in train_good_traces:
            scl.partial_fit(trace[columns])

        
        # Apply the normalization
        for ti, trace in enumerate(traces_data):
            traces_data[ti][columns] = scl.transform(trace[columns])

        for ti, trace in enumerate(train_good_traces):
            train_good_traces[ti][columns] = scl.transform(trace[columns])

        for ti, trace in enumerate(good_witnesses_ea):
            good_witnesses_ea[ti][columns] = scl.transform(trace[columns])

        for ti, trace in enumerate(bad_witnesses_ea):
            bad_witnesses_ea[ti][columns] = scl.transform(trace[columns])

        scl.feature_names_in_ = columns
        
        return traces_data, scl, good_witnesses_ea, bad_witnesses_ea

    @staticmethod
    def denormalize_constant(scaler, name, value):
        try:
            col_idx=scaler.get_feature_names_out().index(name)
            return value*scaler.data_range_[col_idx] + scaler.data_min_[col_idx]
        except ValueError as e:
            print("Cannot denormalize ", name)
            return value


    @staticmethod
    def get_best_ind(front):
        hypervolumes=[] 
        #compute hypervolumnes
        if len(front) > 0: 
            for sol in front:
                hv = pyhv([np.negative(np.asarray(sol.fitness.values))])
                hypervolumes.append(hv.compute([1,]*len(sol.fitness.values)))

            index_selected_from_pareto = np.argmax(hypervolumes)
            return front[index_selected_from_pareto]
        else:
            return None
        

    @staticmethod
    def cxOnePointLeafBiased(ind1, ind2, termpb):
        """Randomly select crossover point in each individual and exchange each
        subtree with the point as root between each individual.
    
        :param ind1: First typed tree participating in the crossover.
        :param ind2: Second typed tree participating in the crossover.
        :param termpb: The probability of choosing a terminal node (leaf).
        :returns: A tuple of two typed trees.
    
        When the nodes are strongly typed, the operator makes sure the
        second node type corresponds to the first node type.
    
        The parameter *termpb* sets the probability to choose between a terminal
        or non-terminal crossover point. For instance, as defined by Koza, non-
        terminal primitives are selected for 90% of the crossover points, and
        terminals for 10%, so *termpb* should be set to 0.1.
        """
    
        if len(ind1) < 2 or len(ind2) < 2:
            # No crossover on single node tree
            return ind1, ind2
    
        # Determine whether to keep terminals or primitives for each individual
        terminal_op = partial(eq, 0)
        primitive_op = partial(lt, 0)
        arity_op1 = terminal_op if random.random() < termpb else primitive_op
        arity_op2 = terminal_op if random.random() < termpb else primitive_op
    
        # List all available primitive or terminal types in each individual
        types1 = defaultdict(list)
        types2 = defaultdict(list)
    
        for idx, node in enumerate(ind1[1:], 1):
            if arity_op1(node.arity):
                types1[node.ret].append(idx)
    
        for idx, node in enumerate(ind2[1:], 1):
            if arity_op2(node.arity):
                types2[node.ret].append(idx)
    
        common_types = list(set(types1.keys()).intersection(set(types2.keys())))
        str_common_types = [str(x) for x in common_types]
        common_types = [x for _, x in sorted(zip(str_common_types, common_types))]

    
        if len(common_types) > 0:
            # Set does not support indexing
            type_ = random.sample(common_types, 1)[0]
            index1 = random.choice(types1[type_])
            index2 = random.choice(types2[type_])
    
            slice1 = ind1.searchSubtree(index1)
            slice2 = ind2.searchSubtree(index2)
            ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
    
        return ind1, ind2
        
    
    ################################## end static methods

    ################################## instance methods
    
    def gen_split_window(self):
        return int(random.uniform(1, self.t_len-1))

    def gen_split_window_h(self, h):
        return int(random.uniform(h, self.t_len-1))
    
    def gen_mult(self, max_val):
        return random.uniform(0, max_val)
    
    '''
    def get_horizon(self, individual): 
        formula = self.ind_to_formula(individual)

        spec = rtamt.StlDenseTimeOfflineSpecification()
        for c in GP_STL.signal_columns:
            spec.declare_var(c, GP_STL.column_types[c])

        spec.spec=formula
        
        try:
            spec.parse()
            
            h = StlHorizon()
            for s in spec.ast.specs:
                horizon = h.visit(s, None)

        except rtamt.STLParseException as e:
            print('STLParseException for formula ', formula, ':', e)
            assert False

        # print(str(individual), int(horizon)/dummy_c)
        
        return math.ceil(horizon) if horizon > 1. else 1
    '''

    @staticmethod
    def get_relative_horizon(individual): ## relative horizon
        
        dummy_c = 9999999999999999999999 ## because rtamt compute only int horizon
        tree = gp.PrimitiveTree(copy.deepcopy(individual))
        formula = GP_STL.get_stl_expr(GP_STL.translate_tree(str(tree), GP_STL.signal_columns), dummy_c)
        # self.ind_to_formula(individual, t_len=t_len)

        spec = rtamt.StlDenseTimeOfflineSpecification()
        for c in GP_STL.signal_columns:
            spec.declare_var(c, GP_STL.column_types[c])

        spec.spec=formula
        
        try:
            spec.parse()
            
            h = StlHorizon()
            for s in spec.ast.specs:
                horizon = h.visit(s, None)

        except rtamt.STLParseException as e:
            print('STLParseException for formula ', formula, ':', e)
            return -1.

        # print(str(individual), int(horizon)/dummy_c)
        
        return int(horizon)/dummy_c

    
    @staticmethod
    def divide_traces_data(traces_data, window, horizon=0):
        traces_pos=[]
        traces_neg=[]
        
        n_augs=len(traces_data)
        
#         print(traces_data[0])
        t_len = len(traces_data[0])   

    
    
        time_pos = list(range(window)) #if window > 1 else [0,0]
        
        neg_len = t_len-window+(horizon) if horizon > 0 else t_len-window
        time_neg = list(range(neg_len)) #if neg_len >1 else [0,0]
        
        
        
        for i, tr in enumerate(traces_data):

            trace_data_pos = {}
            trace_data_neg = {}

            trace_data_pos['time'] = time_pos
            trace_data_neg['time'] = time_neg
            trace_data_pos.update({c: tr[c].iloc[t_len-window:].values.tolist() for c in GP_STL.signal_columns }) # inplace
            trace_data_neg.update({c: tr[c].iloc[0:neg_len].values.tolist() for c in GP_STL.signal_columns }) 

            traces_pos+=[trace_data_pos]            
            if i < n_augs-GP_STL.n_train_good_traces: # debug
                traces_neg+=[trace_data_neg]
                

        
        return traces_pos, traces_neg







    @staticmethod
    def aux_evalIndividual(good_wit, bad_wit, horizon, spec, formula, window, verbose):
        # good witness traces, formula must evaluate to false on them
        how_many_used = 0.
        if len(good_wit) > 0:
            FP = 0.

            for trace in good_wit:

                if len(trace) > horizon:
                    
                    trace_data = {}
                    trace_data['time'] = range(len(trace)) #if len(trace)>1 else [0,0]
                    trace_data.update({c:trace[c].values.copy() for c in GP_STL.signal_columns})
             
                    try:
                        rob=np.array([r[1] for r in evaluate_spec(spec, trace_data)])
                    except Exception as e:
                        print("Traces good witnesses evaluation error for:", formula, 'split w:', window, 't_len', len(trace_data['time']), trace_data, '\nException:', e)
                        assert False
    
                    #if horizon > 1:
                    if horizon >= 1:
                        rob=rob[:-horizon]

                    how_many_used += 1
        
                    if sum(np.isfinite(rob)) == 0:
                        print('Rob not finite for formula:', formula, '\nraw:', str(individual), '\nsplit nw', window, 'h', horizon, 't_len', t_len, 
                              'rel h', GP_STL.get_relative_horizon(individual))
                        assert False 
        
                    max_rob = np.nanmax(rob[np.isfinite(rob)])
                    assert max_rob >= -1 and max_rob <= 1, max_rob
                    FP += 1. if max_rob>=0 else 0.
                    
                    if verbose:
                        for c in GP_STL.signal_columns:
                            plt.plot(trace_data[c], label=c)
                        plt.legend()
                        plt.show()
                        print('Monitoring STL formula:', formula, ' | Max robustness:', max_rob, " | FP:", FP)
                
            
        else:
            print("THIS SHALL NOT HAPPEN")
            FP = 0.
        TN = how_many_used - FP


        # BAD witness traces, formula must evaluate to false on them
        how_many_used = 0.
        if len(bad_wit) > 0:
            TP = 0.

            for trace in bad_wit:

                if len(trace) > horizon:
                    
                    trace_data = {}
                    trace_data['time'] = range(len(trace)) #if len(trace)>1 else [0,0]
                    trace_data.update({c:trace[c].values.copy() for c in GP_STL.signal_columns})
             
                    try:
                        rob=np.array([r[1] for r in evaluate_spec(spec, trace_data)])
                    except Exception as e:
                        print("Traces bad witnesses evaluation error for:", formula, 'split w:', window, 't_len', len(trace_data['time']), trace_data, '\nException:', e)
                        assert False
    
                    #if horizon > 1:
                    if horizon >= 1:
                        rob=rob[:-horizon]

                    how_many_used += 1
        
                    if sum(np.isfinite(rob)) == 0:
                        print('Rob not finite for formula:', formula, '\nraw:', str(individual), '\nsplit nw', window, 'h', horizon, 't_len', t_len, 
                              'rel h', GP_STL.get_relative_horizon(individual))
                        assert False 
        
                    max_rob = np.nanmax(rob[np.isfinite(rob)])
                    assert max_rob >= -1 and max_rob <= 1, max_rob
                    TP += 1. if max_rob>=0 else 0.
                    
                    if verbose:
                        for c in GP_STL.signal_columns:
                            plt.plot(trace_data[c], label=c)
                        plt.legend()
                        plt.show()
                        print('Monitoring STL formula:', formula, ' | Max robustness:', max_rob, " | TP:", TP)

        else:
            TP = 0.
        FN = how_many_used - TP

        return TP, TN, FP, FN
    
    
    # Now we define the evaluation function for each individual
    @staticmethod
    def evalIndividual(i_individual, verbose=False):

        individual = GP_STL.temp[i_individual]

        # check if the individual is feasible
        if not GP_STL.feasible(individual):
            print('This should not happen')
            assert False
            # return 0.,0.#,0.
        
        
        t_len = len(GP_STL.traces_data[0])
        window = individual.window
        ind_height = gp.PrimitiveTree(individual).height
        
        formula = GP_STL.ind_to_formula(individual)
        
        spec = rtamt.StlDiscreteTimeOfflineSpecification()
        for c in GP_STL.signal_columns:
            spec.declare_var(c, GP_STL.column_types[c])

        spec.spec=formula

        try:
            spec.parse()
        except rtamt.STLParseException as e:
            print('STLParseException for formula ', formula, ':', e)
            assert False

        horizon = int(GP_STL.get_horizon_from_mon(spec))


        fitness_acc, fitness_rob = 0., 0.
        traces_pos, traces_neg = GP_STL.divide_traces_data(GP_STL.traces_data, window, horizon)

        assert not horizon > len(traces_neg[0]['time']) and not horizon > len(traces_pos[0]['time'])

        
        # traces_pos
        fitness_rob_pos = 0.
        for i, trace_data in enumerate(traces_pos):
            try:
                rob=evaluate_spec(spec, trace_data)[0][1]
            except Exception as e:
                print("Traces pos evaluation error for:", formula, 'w:', window, 't_len', len(trace_data['time']), trace_data, '\nException:', e)
                assert False

            if rob == -np.inf or rob == np.inf:
                assert False

            # if the formula does not sat the original failure trace, penalize rob and acc and skip evaluation of agumented pos
            if i == 0 and rob < 0:
                assert rob >= -1 and rob <= 1
                fitness_acc += 0
                fitness_rob_pos = -1
                break
            fitness_acc += 1 if rob>=0 else 0
            assert rob >= -1 and rob <= 1
            fitness_rob_pos += rob # robustness at 0

            if verbose:
                for c in GP_STL.signal_columns:
                    plt.plot(trace_data[c], label=c)
                plt.legend()
                plt.show()
                print('Monitoring STL formula:', formula, ' | Robustness:', rob, " | fitness_acc:", fitness_acc)


        # traces_neg
        fitness_rob_neg = 0.
        for i, trace_data in enumerate(traces_neg[:(-GP_STL.n_train_good_traces if GP_STL.n_train_good_traces > 0 else len(traces_neg))]):                

            try:
                rob=np.array([r[1] for r in evaluate_spec(spec, trace_data)])
            except Exception as e:
                print("Traces neg evaluation error for:", formula, 'split w:', window, 't_len', len(trace_data['time']), trace_data, '\nException:', e)
                assert False
            #if horizon > 1:
            if horizon >= 1:
                rob=rob[:-horizon]

            if sum(np.isfinite(rob)) == 0:
                print('Rob not finite for formula:', formula, '\nraw:', str(individual), '\nsplit nw', window, 'h', horizon, 't_len', t_len, 
                      'rel h', GP_STL.get_relative_horizon(individual))
                assert False 

            max_rob = np.nanmax(rob[np.isfinite(rob)])
            # if the formula does not sat the original failure trace, penalize rob and acc and skip evaluation of agumented pos
            if i == 0 and max_rob >= 0:
                assert max_rob >= -1 and max_rob <= 1
                fitness_acc += 0
                fitness_rob_neg = -1
                break
            fitness_acc += 0. if max_rob>=0 else 1.
            assert max_rob >= -1 and max_rob <= 1
            fitness_rob_neg -= max_rob # inverted sign on good traces
        
            
            if verbose:
                for c in GP_STL.signal_columns:
                    plt.plot(trace_data[c], label=c)
                plt.legend()
                plt.show()
                print('Monitoring STL formula:', formula, ' | Max robustness:', max_rob, " | fitness_acc:", fitness_acc)


        if GP_STL.maxfar == 1. :
            return_score_ea =  1
        else:
            assert GP_STL.ea_score_method == "FAR" or (GP_STL.ea_score_method == "MCC" and (len(GP_STL.good_witnesses_ea) > 0 and len(GP_STL.bad_witnesses_ea) > 0))

            # For the fitness used during the evolutionary process
            TP, TN, FP, FN = GP_STL.aux_evalIndividual(GP_STL.good_witnesses_ea, GP_STL.bad_witnesses_ea, horizon, spec, formula, window, verbose)
    
            numerator = (TP * TN) - (FP * FN)
            denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            MCC = numerator / denominator if denominator != 0 else 0
            MCC_norm = (MCC + 1.) / 2.
            
            FAR = FP/(FP + TN) if (FP + TN) > 0 else 0.

            if GP_STL.ea_score_method == "FAR":
                return_score_ea =  1-FAR
            elif GP_STL.ea_score_method == "MCC":
                return_score_ea =  MCC_norm


            # For the early stopping, where the reference set must not change among the generations
            TP, TN, FP, FN = GP_STL.aux_evalIndividual(GP_STL.hv_calc_good_witnesses_ea, GP_STL.hv_calc_bad_witnesses_ea, horizon, spec, formula, window, verbose)
    
            numerator = (TP * TN) - (FP * FN)
            denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            MCC = numerator / denominator if denominator != 0 else 0
            MCC_norm = (MCC + 1.) / 2.
            
            FAR = FP/(FP + TN) if (FP + TN) > 0 else 0.

            if GP_STL.ea_score_method == "FAR":
                return_score_ea_HV =  1-FAR
            elif GP_STL.ea_score_method == "MCC":
                return_score_ea_HV =  MCC_norm
        
        
        n_units = len(traces_pos) + len(traces_neg)
        n_neg_units = len(traces_neg)


        # Combine the two robustness values harmonically
        fitness_rob_pos = fitness_rob_pos/n_units
        fitness_rob_neg = fitness_rob_neg/n_units

        fitness_rob_pos = (fitness_rob_pos + 1) / 2
        fitness_rob_neg = (fitness_rob_neg + 1) / 2
        
        assert fitness_rob_pos >= 0 and fitness_rob_pos <= 1, fitness_rob_pos
        assert fitness_rob_neg >= 0 and fitness_rob_neg <= 1, fitness_rob_neg


        if GP_STL.margin_weight <= 1:
            fitness_rob = min(fitness_rob_pos, fitness_rob_neg*GP_STL.margin_weight)
        else:
            fitness_rob = min(fitness_rob_pos*(1/GP_STL.margin_weight), fitness_rob_neg)

        
        #fitness_rob = (2*np.abs(fitness_rob_pos) * np.abs(fitness_rob_neg)) / (np.abs(fitness_rob_pos) + np.abs(fitness_rob_neg))
        #fitness_rob_pos = (fitness_rob_pos + 1)/2
        #fitness_rob_neg = (fitness_rob_neg + 1)/2
        #fitness_rob = (2*fitness_rob_pos * fitness_rob_neg) / (fitness_rob_pos + fitness_rob_neg)
        
        assert fitness_acc/n_units >=0 and fitness_acc/n_units <= 1 
        assert fitness_rob >= 0 and fitness_rob <= 1, str(fitness_rob) + "_" + str(fitness_rob_pos) + "_" + str(fitness_rob_neg)
        assert ind_height <= GP_STL.max_height
        assert individual.window <= GP_STL.reverse_obj_window
        
        return fitness_acc/n_units, horizon, window, fitness_acc/n_units, fitness_rob, (GP_STL.reverse_obj_window-individual.window)/GP_STL.reverse_obj_window, (GP_STL.max_height-ind_height)/GP_STL.max_height, return_score_ea, return_score_ea_HV




    # Now we define the evaluation function for each individual
    @staticmethod
    def evalIndividual_no_HV_no_trace(i_individual, verbose=False):

        individual = GP_STL.temp[i_individual]

        # check if the individual is feasible
        if not GP_STL.feasible(individual):
            print('This should not happen')
            assert False
            # return 0.,0.#,0.
        
        
        t_len = len(GP_STL.traces_data[0])
        window = individual.window
        ind_height = gp.PrimitiveTree(individual).height
        
        formula = GP_STL.ind_to_formula(individual)
        
        spec = rtamt.StlDiscreteTimeOfflineSpecification()
        for c in GP_STL.signal_columns:
            spec.declare_var(c, GP_STL.column_types[c])

        spec.spec=formula

        try:
            spec.parse()
        except rtamt.STLParseException as e:
            print('STLParseException for formula ', formula, ':', e)
            assert False

        horizon = int(GP_STL.get_horizon_from_mon(spec))


        if GP_STL.maxfar == 1. :
            return_score_ea =  1
        else:
            assert GP_STL.ea_score_method == "FAR" or (GP_STL.ea_score_method == "MCC" and (len(GP_STL.good_witnesses_ea) > 0 and len(GP_STL.bad_witnesses_ea) > 0))

            # For the fitness used during the evolutionary process
            TP, TN, FP, FN = GP_STL.aux_evalIndividual(GP_STL.good_witnesses_ea, GP_STL.bad_witnesses_ea, horizon, spec, formula, window, verbose)
    
            numerator = (TP * TN) - (FP * FN)
            denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            MCC = numerator / denominator if denominator != 0 else 0
            MCC_norm = (MCC + 1.) / 2.
            
            FAR = FP/(FP + TN) if (FP + TN) > 0 else 0.

            if GP_STL.ea_score_method == "FAR":
                return_score_ea =  1-FAR
            elif GP_STL.ea_score_method == "MCC":
                return_score_ea =  MCC_norm
        
        return return_score_ea






    

    @staticmethod
    def ind_to_formula(individual, scaler=None):
        tree = gp.PrimitiveTree(copy.deepcopy(individual))
        return GP_STL.get_stl_expr(GP_STL.translate_tree(str(tree), GP_STL.signal_columns), individual.bound_mult, scaler)
    





    
    def run_gp_stl(self, traces_data, max_gen=500, pop_size=100, cxpb=0.7, mutpb=0.5, verbose=False, alg_seed=42, train_good_traces=[],good_witnesses_ea=[],bad_witnesses_ea=[], maxfar=0., minacc=0.5, return_multiple_formulas_ea=False):
        print("Number of (augmented) good witnesses:", len(good_witnesses_ea))
        print("Number of (augmented) bad witnesses:", len(bad_witnesses_ea))
        print("Maximum number of false positive traces allowed for a solution:", GP_STL.num_good_witnesses*maxfar)

        pops = []
        fronts = {}


        np.random.seed(alg_seed)
        random.seed(alg_seed)

        timeout=-1
        start_time=time.time()
        
        self.train_good_traces = train_good_traces
        GP_STL.n_train_good_traces = len(train_good_traces)

        assert GP_STL.reverse_obj_window is not None

        GP_STL.maxfar = maxfar

        if GP_STL.goodbad_traces_at_ea_generation > 0:
            good_witnesses_ea_original = copy.deepcopy(good_witnesses_ea)
            bad_witnesses_ea_original = copy.deepcopy(bad_witnesses_ea)
            GP_STL.traces_data, scl, good_witnesses_ea_original, bad_witnesses_ea_original = GP_STL.normalize_traces_data(traces_data, GP_STL.signal_columns, self.train_good_traces, good_witnesses_ea_original, bad_witnesses_ea_original)
            GP_STL.hv_calc_good_witnesses_ea = random.sample(good_witnesses_ea_original, min(max(10, GP_STL.num_good_witnesses), 2000)) 
            GP_STL.hv_calc_bad_witnesses_ea = [] # For future versions
            print("Number of reference good/bad witnesses for HV calculation:", len(GP_STL.hv_calc_good_witnesses_ea), len(GP_STL.hv_calc_bad_witnesses_ea))
        else:      
            GP_STL.traces_data, scl, GP_STL.good_witnesses_ea, GP_STL.bad_witnesses_ea = GP_STL.normalize_traces_data(traces_data, GP_STL.signal_columns, self.train_good_traces, good_witnesses_ea, bad_witnesses_ea)


        list_training_hypervolume = []
        
        n_units = len(GP_STL.traces_data)
        t_len = len(GP_STL.traces_data[0])
        
        self.t_len = t_len
        
        
        # Find optimal formulae through GENETIC Programming algorithm
        if verbose:
            print('Running stl inference on...', n_units, 'units of length ', len(GP_STL.traces_data[0]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
    
            # Variable keeping track of the number of generation
            g = 0

            if GP_STL.goodbad_traces_at_ea_generation > 0:
                GP_STL.good_witnesses_ea = random.sample(good_witnesses_ea_original, GP_STL.num_good_witnesses) #if (len(good_witnesses_ea) > 0 and GP_STL.num_good_witnesses > 0) else []
                GP_STL.bad_witnesses_ea = random.sample(bad_witnesses_ea_original, GP_STL.num_bad_witnesses) #if (len(bad_witnesses_ea) > 0 and GP_STL.num_bad_witnesses > 0) else []
                print("Rotating with good/bad traces:", len(good_witnesses_ea_original), len(GP_STL.good_witnesses_ea), len(bad_witnesses_ea_original), len(GP_STL.bad_witnesses_ea))


            pop=[]
            # Generate the population
            gen_seeds=np.random.randint(low=0, high=4294967295, size=(GP_STL.max_gen_height-GP_STL.min_height+1), dtype='uint32') 
            for i,j in enumerate(range(GP_STL.min_height, GP_STL.max_gen_height+1)):
                random.seed(gen_seeds[i])
                np.random.seed(gen_seeds[i])
                pop += getattr(self.toolbox, "population"+str(j))(n=int(pop_size/len(gen_seeds)))

                

            
               
            for p in pop:
                p.window = self.gen_split_window()
                relative_horizon = GP_STL.get_relative_horizon(p)

                max_h = min(p.window, self.max_horizon, self.t_len - p.window)
                relative_horizon = relative_horizon if relative_horizon >= 1/max_h else max_h                
                p.bound_mult = self.gen_mult(max_h / relative_horizon)

            
                                  
            if verbose:
                print('== Finished to generate the population ==')
                for p in pop:
                    print (self.ind_to_formula(p), 'height:', p.height, 'w:', p.window)#, 'h:', self.get_horizon(p) )
                    print('----------------------------------------------------')


            if(GP_STL.processes>1):
                GP_STL.temp = pop
                
                p=ProcessPool(ncpus=GP_STL.processes)
                p.restart()
                fitnesses = p.map(self.evalIndividual, [i for i in range(len(GP_STL.temp))])
                p.close()
                p.join()       
            else:
                fitnesses = [self.toolbox.evaluate(par) for par in pop]

            
            fitnesses_hv = []
            for ind, fit in zip(pop, fitnesses):
                assert np.isfinite(fit[0]) and np.isfinite(fit[4])
                assert fit[3] == fit[0]
                assert np.nan_to_num(fit[0],0) >= 0 and np.nan_to_num(fit[0],0) <= 1
                assert np.nan_to_num(fit[4],0) >= 0 and np.nan_to_num(fit[4],0) <= 1
                assert fit[5] >= 0 and fit[5] <= 1
                assert fit[6] >= 0 and fit[6] <= 1
                assert fit[7] >= 0 and fit[7] <= 1
                assert fit[8] >= 0 and fit[8] <= 1
                
                ind.fitness.values = (np.nan_to_num(fit[0],0), np.nan_to_num(fit[4],0), fit[7])
                ind.h = np.nan_to_num(fit[1],0)
                ind.abs_w = np.nan_to_num(fit[2],0)
                ind.accuracy = np.nan_to_num(fit[3],0)
                ind.robustness = np.nan_to_num(fit[4],0)
                ind.hv_score = fit[8]

                if GP_STL.goodbad_traces_at_ea_generation > 0: 
                    fitnesses_hv.append((np.nan_to_num(fit[0],0), np.nan_to_num(fit[4],0), fit[8]))
                else:
                    fitnesses_hv.append((np.nan_to_num(fit[0],0), np.nan_to_num(fit[4],0), fit[7]))


            
            if(GP_STL.processes>1):
                GP_STL.temp = fitnesses_hv
                p = multiprocessing.Pool(GP_STL.processes)
                is_dom_list_train = p.map(parallel_extract_all_front, [fitness for fitness in fitnesses_hv])
                p.close()
                p.join()
            else:
                is_dom_list_train = [parallel_extract_all_front([fitness, fitnesses_hv]) for fitness in fitnesses_hv]

            try :
                non_dom = [x.fitness.values for i, x in enumerate(pop) if not is_dom_list_train[i]]
                hv = pyhv([np.negative(np.asarray(x)) for x in non_dom])
                hv = hv.compute([1,1,1]) 
                list_training_hypervolume.append(hv)   
            except Exception as e:
                print (i, len(is_dom_list_train), len(pop))


            
            front = [x for i, x in enumerate(pop) if not is_dom_list_train[i]]
            # Removing duplicates from the front    
            uniquefront = {}
            for sol in front:
                w = sol.window
                uniquefront[str(sol)+'|'+str(w)] = sol
                fronts[str(sol)+'|'+str(w)] = self.toolbox.clone(sol)
            front = list(uniquefront.values())


            pops.append(list(map(self.toolbox.clone, pop)))
            

            fitnesses = [f[0] for f in fitnesses]
            min_fit = np.min(fitnesses)
            max_fit = np.max(fitnesses)
            mean_fit = np.mean(fitnesses)
            best_ind = GP_STL.get_best_ind(front)
            best_ind_formula = self.ind_to_formula(best_ind)


            list_training_info =  [min_fit, max_fit, mean_fit, best_ind.h, best_ind.height, best_ind.abs_w, best_ind.accuracy, best_ind.robustness]           
            

            if verbose:
                
                plt.figure(figsize=(6,3))
                plt.scatter([x.fitness.values[0] for x in pop], [x.fitness.values[1] for x in pop], color='tab:blue')
                plt.scatter([x.fitness.values[0] for i, x in enumerate(pop) if not is_dom_list_train[i]], [x.fitness.values[1] for i, x in enumerate(pop) if not is_dom_list_train[i]], color='tab:red')
                plt.xlabel("Accuracy")
                plt.ylabel("Robustness")
                plt.show()
                
                
                # More detailed info -----------------------------------------------------------------
                
                # Determine hypervolume                
                print('fitness min:', min_fit, 'fitness max:',  max_fit, 'fitness avg:', mean_fit)
                print('best formula:', best_ind_formula, '\nhorizon:', best_ind.h, 'height:', best_ind.height, 'window:', best_ind.abs_w)
                print('--------------------------------------------------')
                
                
                plt.figure(figsize=(15,3))

                plt.subplot(1,4,1)
                plt.hist(fitnesses)
                plt.xlabel("Fitness")
                plt.ylabel("Freq")

                plt.subplot(1,4,2)
                plt.hist([ind.h for ind in pop])
                plt.xlabel("Horizon")
                plt.ylabel("Freq")

                plt.subplot(1,4,3)
                plt.hist([ind.height for ind in pop])
                plt.xlabel("Height")
                plt.ylabel("Freq")

                plt.subplot(1,4,4)
                plt.hist([ind.abs_w for ind in pop])
                plt.xlabel("Window")
                plt.ylabel("Freq")

                plt.show()
            
                
                plt.subplot(1,2,1)
                plt.hist([ind.accuracy for ind in pop])
                plt.xlabel("Acc")
                plt.ylabel("Freq")

                plt.subplot(1,2,2)
                plt.hist([ind.robustness for ind in pop])
                plt.xlabel("Rob")
                plt.ylabel("Freq")
            
    

            # Selection step
            pop = self.toolbox.select(pop, len(pop))


            
            # Evolve the population
            for g in tqdm(range(1, max_gen)):

                if verbose:
                    it_time = time.time()
                    print("-- Generation %i --" % g)

                
                # if good and bad traces are choosen at each generation, first, we need to calculate again the fitnees on all the population
                if (GP_STL.goodbad_traces_at_ea_generation > 0) and (g%GP_STL.goodbad_traces_at_ea_generation == 0):
                    #print("Changing reference good witnesses...")
                    GP_STL.good_witnesses_ea = random.sample(good_witnesses_ea_original, GP_STL.num_good_witnesses) #if (len(good_witnesses_ea) > 0 and GP_STL.num_good_witnesses > 0) else []
                    GP_STL.bad_witnesses_ea = random.sample(bad_witnesses_ea_original, GP_STL.num_bad_witnesses) #if (len(bad_witnesses_ea) > 0 and GP_STL.num_bad_witnesses > 0) else []


                    if(GP_STL.processes>1):
                        GP_STL.temp = pop
                        
                        p=ProcessPool(ncpus=GP_STL.processes)
                        p.restart()
                        fitnesses = p.map(self.evalIndividual_no_HV_no_trace, [i for i in range(len(GP_STL.temp))])
                        p.close()
                        p.join()
                    else:
                        fitnesses = [self.toolbox.evaluate(par) for par in pop]
                        
    
                    for ind, fit in zip(pop, fitnesses):
                        assert fit >= 0 and fit <= 1
                        
                        ind.fitness.values = (ind.fitness.values[0], ind.fitness.values[1], fit)

                

                # Clone the selected individuals: this ensures that we don’t use a reference to the individuals but a completely independent instance
                offspring = list(map(self.toolbox.clone, pop))


                # Apply crossover and mutation on the offspring
                
                # Crossover
                for i in range(1, len(offspring), 2): 
                    if random.random() < cxpb:
                        rolled = random.random()
                        tmp_seed_random = random.randint(0, 999999)
                        tmp_seed_random_numpy = np.random.randint(999999, size=10)[3]
                        mut1, mut2 = self.toolbox.mate(ind1=self.toolbox.clone(offspring[i-1]), ind2=self.toolbox.clone(offspring[i]), roll=rolled)
                        random.seed(tmp_seed_random)
                        np.random.seed(tmp_seed_random_numpy)

                        
                        if GP_STL.feasible(mut1):
                            relative_horizon = GP_STL.get_relative_horizon(mut1)
                            max_h = min(mut1.window, self.max_horizon, self.t_len - mut1.window)
                            relative_horizon = relative_horizon if relative_horizon >= 1/max_h else max_h                
                            mut1.bound_mult = self.gen_mult(max_h / relative_horizon)
            
                            offspring[i-1] = mut1
                            del offspring[i-1].fitness.values
                        if GP_STL.feasible(mut2):
                            relative_horizon = GP_STL.get_relative_horizon(mut2)
                            max_h = min(mut2.window, self.max_horizon, self.t_len - mut2.window)
                            relative_horizon = relative_horizon if relative_horizon >= 1/max_h else max_h                
                            mut2.bound_mult = self.gen_mult(max_h / relative_horizon)
                            
                            offspring[i] = mut2
                            del offspring[i].fitness.values

                # Mutation 
                for mi,mutant in enumerate(offspring):
                    if random.random() < mutpb / g**(1/GP_STL.mutation_decay): ## N.B.:  0 <= random() < 1.0
                        mut = self.toolbox.mutate(individual=self.toolbox.clone(mutant), roll=random.random())[0]
                                               
                        if GP_STL.feasible(mut) and mut.height <= GP_STL.max_height:
                            
                            relative_horizon = GP_STL.get_relative_horizon(mut)
                            max_h = min(mut.window, self.max_horizon, self.t_len - mut.window)
                            relative_horizon = relative_horizon if relative_horizon >= 1/max_h else max_h                
                            mut.bound_mult = self.gen_mult(max_h / relative_horizon)
                                
                            offspring[mi] = mut
                            del offspring[mi].fitness.values 
   
                        
                
                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]


                if verbose:
                    print('== Invalid individuals ==')
                    for p in invalid_ind:
                        print (self.ind_to_formula(p), 'height:', p.height, 'w:', p.window)#, 'h:', self.get_horizon(p) )
                        print('----------------------------------------------------')


                
                if(GP_STL.processes>1):
                    GP_STL.temp = invalid_ind
                    
                    p=ProcessPool(ncpus=GP_STL.processes)
                    p.restart()
                    fitnesses = p.map(self.evalIndividual, [i for i in range(len(GP_STL.temp))])
                    p.close()
                    p.join()
                else:
                    fitnesses = [self.toolbox.evaluate(par) for par in invalid_ind]
                    

                for ind, fit in zip(invalid_ind, fitnesses):
                    assert np.isfinite(fit[0]) and np.isfinite(fit[4])
                    assert fit[0] == fit[3]
                    assert np.nan_to_num(fit[0],0) >= 0 and np.nan_to_num(fit[0],0) <= 1
                    assert np.nan_to_num(fit[4],0) >= 0 and np.nan_to_num(fit[4],0) <= 1
                    assert fit[5] >= 0 and fit[5] <= 1
                    assert fit[6] >= 0 and fit[6] <= 1
                    assert fit[7] >= 0 and fit[7] <= 1
                    assert fit[8] >= 0 and fit[8] <= 1
                    
                    ind.fitness.values = (np.nan_to_num(fit[0],0), np.nan_to_num(fit[4],0), fit[7])
                    ind.h = np.nan_to_num(fit[1],0)
                    ind.abs_w = np.nan_to_num(fit[2],0)
                    ind.accuracy = np.nan_to_num(fit[3],0)
                    ind.robustness = np.nan_to_num(fit[4],0)
                    ind.hv_score = fit[8]



                    
                
                # Select the next generation individuals, having the same size as the previous generation
                offspring = self.toolbox.select(pop + offspring, len(pop))

                # Finally, replace the old population by the offspring, composed of some elements of the old population and some new elements
                pop[:] = offspring


                # Determine hypervolume
                if GP_STL.goodbad_traces_at_ea_generation > 0: 
                    fitnesses_hv = []
                    for ind in pop:
                        fitnesses_hv.append((ind.fitness.values[0], ind.fitness.values[1], ind.hv_score))
                else:
                    fitnesses_hv = [ind.fitness.values for ind in pop]

                if(GP_STL.processes>1):
                    GP_STL.temp = fitnesses_hv
                    p = multiprocessing.Pool(GP_STL.processes)
                    is_dom_list_train = p.map(parallel_extract_all_front, [fitness for fitness in fitnesses_hv])
                    p.close()
                    p.join()
                else:
                    is_dom_list_train = [parallel_extract_all_front([fitness, fitnesses_hv]) for fitness in fitnesses_hv]

                non_dom = [x.fitness.values for i, x in enumerate(pop) if not is_dom_list_train[i]]
                hv = pyhv([np.negative(np.asarray(x)) for x in non_dom])
                hv = hv.compute([1,1,1]) # ref point for the correlation is 1, since I have negated it
                list_training_hypervolume.append(hv) 

                front = [x for i, x in enumerate(pop) if not is_dom_list_train[i]]
                # Removing duplicates from the front    
                if (GP_STL.goodbad_traces_at_ea_generation > 0) and (g%GP_STL.goodbad_traces_at_ea_generation == (GP_STL.goodbad_traces_at_ea_generation-1)):
                    for sol in front:
                        # if sol.fitness.values[0] > 0.5:
                        w = sol.window
                        fronts[str(sol)+'|'+str(w)] = self.toolbox.clone(sol)            


            

                fitnesses = [ind.fitness.values[0] for ind in pop]

                min_fit = np.min(fitnesses)
                max_fit = np.max(fitnesses)
                mean_fit = np.mean(fitnesses)
 
                best_ind = GP_STL.get_best_ind(front)  
                best_ind_formula = self.ind_to_formula(best_ind)
                
                list_training_info = np.vstack([list_training_info, [min_fit, max_fit, mean_fit, best_ind.h, best_ind.height, best_ind.abs_w, best_ind.accuracy, best_ind.robustness]])


                pops.append(list(map(self.toolbox.clone, pop)))
                
                
                if verbose:
                    
                    
                    # Removing duplicates from the front    
                    uniquefront = {}
                    for sol in front:
                        w = sol.window
                        uniquefront[str(sol)+'|'+str(w)] = sol
                    front = list(uniquefront.values())
                    print('Extractable from size:', len(front))                   
                    
                    print('Generation time:', time.time()-it_time) 
                    
                    print("Training   set hypervolume:", hv)
                    plt.figure(figsize=(12,3))

                    plt.subplot(1,2,1)
                    plt.scatter([x.fitness.values[0] for x in pop], [x.fitness.values[1] for x in pop], color='tab:blue')
                    plt.scatter([x.fitness.values[0] for i, x in enumerate(pop) if not is_dom_list_train[i]], [x.fitness.values[1] for i, x in enumerate(pop) if not is_dom_list_train[i]], color='tab:red')
                    plt.xlabel("Accuracy")
                    plt.ylabel("Robustness")

                    plt.subplot(1,2,2)
                    plt.plot(list(range(len(list_training_hypervolume))), list_training_hypervolume)

                    plt.show()
                    
                    
                    # More detailed info -----------------------------------------------------------------
                    
                    
                    
                    

                    # Removing duplicates from the front
                    print('fitness min:', min_fit, 'fitness max:',  max_fit, 'fitness avg:', mean_fit)
                    print('best formula:', best_ind_formula, '\nhorizon:', best_ind.h, 'height:', best_ind.height, 'window:', best_ind.abs_w)
                    
                    fullmap = {}
                    for ind in pop:
                        fullmap[str(ind)+ '|' + str(ind.window)] = None
                    
                    print('n unique sol:', len(fullmap.keys()))
                    print('--------------------------------------------------')
                    
                    # ---------------------------------------------------------------------------------
                    plt.figure(figsize=(15,3))

                    plt.subplot(1,4,1)
                    plt.hist(fitnesses)
                    plt.xlabel("Fitness")
                    plt.ylabel("Freq")

                    plt.subplot(1,4,2)
                    plt.hist([ind.h for ind in pop])
                    plt.xlabel("Horizon")
                    plt.ylabel("Freq")
                    
                    plt.subplot(1,4,3)
                    plt.hist([ind.height for ind in pop])
                    plt.xlabel("Height")
                    plt.ylabel("Freq")
                    
                    plt.subplot(1,4,4)
                    plt.hist([ind.abs_w for ind in pop])
                    plt.xlabel("Window")
                    plt.ylabel("Freq")

                    plt.show()
                   
                    # -------------------------------
                                        
                    plt.figure(figsize=(15,3))

                    plt.subplot(1,4,1)
                    
                    plt.plot(list_training_info[:, 2])
                    #plt.fill_between(list_training_info[:, 0], list_training_info[:, 1], alpha=.3)
                    plt.yscale('log')
                    plt.xlabel("Generation")
                    plt.ylabel("Mean Fitness")

                    plt.subplot(1,4,2)
                    plt.plot(list_training_info[:, 3])
                    plt.xlabel("Generation")
                    plt.ylabel("Best Ind. Horizon")
                    
                    plt.subplot(1,4,3)
                    plt.plot(list_training_info[:, 4])
                    plt.xlabel("Generation")
                    plt.ylabel("Best Ind. Height")


                    plt.subplot(1,4,4)
                    plt.plot(list_training_info[:, 5])
                    plt.xlabel("Generation")
                    plt.ylabel("Best Ind. Window")

                    plt.show()
                    
                    # -------------------------------------------
                    
                    plt.figure(figsize=(15,3))

                    plt.subplot(1,4,1)
                    plt.hist([ind.accuracy for ind in pop])
                    plt.xlabel("Accuracy")
                    plt.ylabel("Freq")
                        
                    plt.subplot(1,4,2)
                    plt.plot(list_training_info[:, 6])
                    plt.ylim(-0.01,1.01)
                    plt.xlabel("Best Ind. Accuracy")
                    plt.ylabel("Generation")
                    
                    plt.subplot(1,4,3)
                    plt.hist([ind.robustness for ind in pop])
                    plt.xlabel("Robustness")
                    plt.ylabel("Freq")
                    
                    plt.subplot(1,4,4)
                    plt.plot(list_training_info[:, 7])
                    plt.ylim(0.,1.)
                    plt.xlabel("Best Ind.Robustness")
                    plt.ylabel("Generation")

                    plt.show()
 

                # Early stopping conditions
                hypervolume_window = GP_STL.ea_patience
                if g >= hypervolume_window:
                    if np.all(np.asarray(list_training_hypervolume)[g-hypervolume_window] + GP_STL.epsilon >= np.asarray(list_training_hypervolume)[g-hypervolume_window:]):
                        print("Stopping due to low difference in fitness increase.", list_training_hypervolume[g], list_training_hypervolume[g-hypervolume_window])
                        break
        
                if timeout >= 0. and (time.time() - start_time) > timeout:
                    break


            assert len(pops) == len(list_training_hypervolume)
            
            # if good and bad traces are choosen at each generation, at the end of the generations we need also to revaluate the entire pop an all traces
            if GP_STL.goodbad_traces_at_ea_generation > 0:

                GP_STL.good_witnesses_ea = good_witnesses_ea_original if GP_STL.num_good_witnesses > 0 else []
                GP_STL.bad_witnesses_ea = bad_witnesses_ea_original if GP_STL.num_bad_witnesses > 0 else []

                #print("Best HV:", np.argmax(list_training_hypervolume))
                pop = pops[np.argmax(list_training_hypervolume)]
                if return_multiple_formulas_ea:
                    pop.extend(list(fronts.values()))#fronts                

                if(GP_STL.processes>1):
                    GP_STL.temp = pop
                    p=ProcessPool(ncpus=GP_STL.processes)
                    p.restart()
                    fitnesses = p.map(self.evalIndividual_no_HV_no_trace, [i for i in range(len(GP_STL.temp))])
                    p.close()
                    p.join()
                else:
                    fitnesses = [self.toolbox.evaluate(par) for par in pop]
                    

                for ind, fit in zip(pop, fitnesses):
                    assert fit >= 0 and fit <= 1
                    
                    ind.fitness.values = (ind.fitness.values[0], ind.fitness.values[1], fit)

                # Determine hypervolume and front
                fitnesses = [ind.fitness.values for ind in pop]
    
                if(GP_STL.processes>1):
                    GP_STL.temp = fitnesses
                    p = multiprocessing.Pool(GP_STL.processes)
                    is_dom_list_train = p.map(parallel_extract_all_front, [fitness for fitness in fitnesses])
                    p.close()
                    p.join()
                else:
                    is_dom_list_train = [parallel_extract_all_front([fitness, fitnesses]) for fitness in fitnesses]
    
                non_dom = [x.fitness.values for i, x in enumerate(pop) if not is_dom_list_train[i]]
                hv = pyhv([np.negative(np.asarray(x)) for x in non_dom])
                hv = hv.compute([1,1,1])#,1]) # ref point for the correlation is 1, since I have negated it
                list_training_hypervolume.append(hv) 
    
                front = [x for i, x in enumerate(pop) if not is_dom_list_train[i]] 

            
            # Removing duplicates, high far, and low acc formulas from the front
            #print('Remove duplicates from front')
            uniquefront = {}
            for sol in front:
                w = sol.window
                
                if sol.fitness.values[0] > minacc and sol.fitness.values[2] >= 1 - GP_STL.maxfar:
                    uniquefront[str(sol)+'|'+str(w)] = sol
                    
            front = list(uniquefront.values())
            
            
            hypervolumes=[] 
            #compute hypervolumnes
            for sol in front:
                hv = pyhv([np.negative(np.asarray(sol.fitness.values))])
                hypervolumes.append(hv.compute([1,]*len(sol.fitness.values)))
 

            if len(hypervolumes) > 0:
                if return_multiple_formulas_ea:
                    formulas = []
                    windows = []
                    fit1s = []
                    fit2s = []
                    horizons = []
                    formulas_noNorm = []
                    for f in front:
                        formulas.append(self.ind_to_formula(f, scl))
                        windows.append(f.window)
                        fit1s.append(f.fitness.values[0])
                        fit2s.append(1-f.fitness.values[2])
                        horizons.append(f.h)
                        formulas_noNorm.append(self.ind_to_formula(f))
                    return formulas, windows, fit1s, fit2s, horizons, formulas_noNorm, copy.deepcopy(scl)
                else:
                    index_selected_from_pareto = np.argmax(hypervolumes)
                    # print(index_selected_from_pareto)
                    f=front[index_selected_from_pareto]
                    return [self.ind_to_formula(f, scl)], \
                           [f.window], \
                           [f.fitness.values[0]], \
                           [1-f.fitness.values[2]], \
                           [f.h], \
                           [self.ind_to_formula(f)], \
                           copy.deepcopy(scl)
            else:
                return [],[],[],[],[],[],[]

        



## MONITOR utility functions 

# completely random shuffle
def get_shuffled_units_online(source, seed, n=-1):
    us = source.serial_number.unique()[:]
    np.random.seed(seed)
    np.random.shuffle(us)
    if n > 0:
        return us[:n]
    else:
        return us

# here, interleave failures and good traces eavenly, starting with a failure trace
# the first method, given a length of an array to generate and the number of 1s to fill in it (remaining values are assumed to be 0s),
# it generates an array of 0s and 1s as uniformly distributed as possible, always staring with a 1
def generate_eavenly_distributed_array(length, num_ones, start_with_one=True): # otherwise, start with 0 if False
    if length == 1 and num_ones > 0:
        return [1]
    elif length == 1 and num_ones == 0:
        return [0]
    else:
        if start_with_one:
            return generate_eavenly_distributed_array(int(np.ceil(length/2)), int(np.ceil(num_ones/2)), start_with_one) + generate_eavenly_distributed_array(int(np.floor(length/2)), int(np.floor(num_ones/2)), start_with_one)
        else:
            return generate_eavenly_distributed_array(int(np.floor(length/2)), int(np.floor(num_ones/2)), start_with_one) + generate_eavenly_distributed_array(int(np.ceil(length/2)), int(np.ceil(num_ones/2)), start_with_one)

def get_shuffled_units_warmup(source, seed, start_with_failure=True): # otherwise, start with a good trace if False
    random.seed(seed)
    failure_units = source[source.failure == 1].serial_number.unique().tolist()
    good_units = source[source.failure == 0].serial_number.unique().tolist()
    random.shuffle(failure_units)
    random.shuffle(good_units)
    distribution = generate_eavenly_distributed_array(len(good_units)+len(failure_units), len(failure_units), start_with_failure)
    num_good_distr, num_fail_distr = np.unique(distribution, return_counts = True)[1]
    assert num_good_distr == len(good_units) and num_fail_distr == len(failure_units)
    if start_with_failure:
        assert distribution[0] == 1
    else:
        assert distribution[0] == 0
    ret_sequence_units = []
    for el in distribution:
        if el == 1:
            ret_sequence_units.append(failure_units.pop())
        else:
            ret_sequence_units.append(good_units.pop())
    return ret_sequence_units


    


def get_unit_trace(source, unit, discretized=False):
    if discretized:
        return get_discretized_trace(source.query("serial_number==@unit").sort_values('date')).iloc[-max_trace_length:]
    else:
        return source.loc[source.serial_number==unit].sort_values('date').iloc[-max_trace_length:]

def get_discretized_trace(trace):
    disc_t=trace.copy()
    for c in sensor_columns:
        disc_t[c] = [ c + '_' + v for v in list(ts_to_string(get_normalized_ts(disc_t[c], feats_means[c], feats_stddevs[c]), cuts_for_asize(feats_levels[c]))) ]
        disc_t[c] = disc_t[c].astype('category')
    return disc_t


def get_normalized_ts(ts, mean, stddev):
    return (ts.values-mean)/stddev



        
def get_augmented_traces(source, unit, t, seed, n_augs, sensor_columns, noise_scale=0.01, noise_kind='additive', col_map_std={}): # BEWARE: different from the LTL version


    crop_list_perc = [t-int(t*0.02), t-int(t*0.05), t-int(t*0.10), t-int(t*0.20), t-int(t*0.30)]
    
    traces_data=[]
    
    trace = source.loc[source.serial_number==unit].sort_values('date').iloc[-max_trace_length:]

    disc_t = trace.copy()

    traces_data += [disc_t.iloc[0:t+1]]

    sensors_to_plot=[] # sensor_columns

    if len(sensors_to_plot)>0:
        plot(trace[sensors_to_plot[0]].values.T)

    for i in range(n_augs):             

        unit_data = trace.copy()
        unit_ts=unit_data[sensor_columns].values

        if noise_kind == 'additive':
            augmenter = (
                 AddNoise(distr='gaussian', scale=noise_scale, kind=noise_kind, seed=seed+int.from_bytes(str(unit).encode('utf-8'), "little")%1000000+i)#np.random.randint(low=0, high=4294967295, size=1, dtype='uint32')) #scale=0.01
            )            
        elif noise_kind == 'ALL':
            # augmenter instance
            augmenter = (
                 AddNoise(distr='gaussian', scale=noise_scale, kind='additive', seed=seed+int.from_bytes(str(unit).encode('utf-8'), "little")%1000000+i) @ 0.8 
                + TimeWarp(seed=seed+int.from_bytes(str(unit).encode('utf-8'), "little")%1000000+i) @ 0.5
            )


        if len(col_map_std) > 0:
            for col in sensor_columns:
                unit_data[col] = np.asarray(unit_data[col].values) + random.uniform(-col_map_std[col], col_map_std[col])
        
        unit_data[sensor_columns] = augmenter.augment(unit_ts.T).T

        if len(sensors_to_plot)>0:
            plot(unit_data[sensors_to_plot[0]].values.T)
            plt.show()
        
        traces_data+=[unit_data.iloc[0:t+1]]
    return traces_data




### Considered METRICS
def f1_score(tp, fp, nn, n_p):
    p=tp/(tp+fp) if (tp+fp) > 0 else 1.
    r=tp/(n_p) if n_p > 0 else 1.
    return (p*r*2)/(p+r) if (p+r) > 0 else 0.


def far_score(tp, fp, nn, n_p):
    return (fp/nn) if nn > 0 else 0.


def get_dt_new_score(score, is_tp, score_beta):
    new_score = 0. if is_tp else 1. 
    return (1.-score_beta) * new_score + score_beta * score





def update_dt_data(dts_pool, dt_idxs, t, trace, is_good, score_beta, max_score_pool, skip_delete=False):
        
    tp_idxs = []
    
    for dt_idx in dt_idxs: 
        
        if not dts_pool.loc[dts_pool.dt_number==dt_idx, 'active'].iloc[0]:
            continue

        
        dt_tp = dts_pool.iloc[dt_idx]['dt_tp'] 
        dt_fp = dts_pool.iloc[dt_idx]['dt_fp']
        dt_nn = dts_pool.iloc[dt_idx]['dt_nn'] 
        dt_np = dts_pool.iloc[dt_idx]['dt_np'] 
        score  = dts_pool.iloc[dt_idx]['score']
        dt_dep_level = dts_pool.iloc[dt_idx]['dep_level']
        dt_dependencies = dts_pool.iloc[dt_idx]['dependency']

        # is false positive
        is_fp = is_good

        if is_good: 
            print('False positive caused by triggered dt')
            


        # false positive
        if is_fp:
            # update dt data
            dt_fp += 1
            dts_pool.loc[dts_pool.dt_number==dt_idx, 'dt_fp'] = dt_fp
            dt_nn += 1
            dts_pool.loc[dts_pool.dt_number==dt_idx, 'dt_nn'] = dt_nn
        else:
            dt_tp += 1
            dts_pool.loc[dts_pool.dt_number==dt_idx, 'dt_tp'] = dt_tp
            dt_np += 1
            dts_pool.loc[dts_pool.dt_number==dt_idx, 'dt_np'] = dt_np
            tp_idxs += [dt_idx]

        new_score = get_dt_new_score(score, not is_fp, score_beta)
        
        dts_pool.loc[dts_pool.dt_number==dt_idx, 'score'] = new_score

        if not skip_delete and new_score > max_score_pool:
            # deactivate dt (and its dependencies)
            print('Deactivating rule for high far')
            deactivate_dt(dts_pool, dt_idx)
        
    return tp_idxs



def update_dt_miss_data(dts_pool, is_good, score_beta, max_score_pool, skip_delete=False): 
        
    if len(dts_pool[(dts_pool.active==True) & (dts_pool.suspended==False)]) > 0: 
        for dt_idx in dts_pool[(dts_pool.active==True) & (dts_pool.suspended==False) ].dt_number.values:
            
            if is_good:
                dts_pool.loc[dts_pool.dt_number==dt_idx, 'dt_nn'] += 1

            new_score = get_dt_new_score(dts_pool.iloc[dt_idx]['score'], True, score_beta)

            dts_pool.loc[dts_pool.dt_number==dt_idx, 'score'] = new_score

            if not skip_delete and new_score > max_score_pool:
                # deactivate dt 
                print('Deactivating rule for high far')
                deactivate_dt(dts_pool, dt_idx)


def deactivate_dt(dts_pool, dt_idx):
    # return 
    if not dts_pool.loc[dts_pool.dt_number==dt_idx, 'active'].iloc[0]:
        return

## Uncomment to deactivate deps
#     f = lambda x: dt_idx in x
#     mask = dts_pool['dependency'].apply(f)
#     dt_deps = dts_pool.loc[mask, 'dt_number'].unique()
    
#     for dt_i in dt_deps:
#         deactivate_dt(dts_pool, dt_i)

    dts_pool.loc[dts_pool.dt_number==dt_idx, 'active'] = False
    print('Rule', dt_idx, 'deactivated, there are ', len(dts_pool[dts_pool.active==True]), 'active rules in the pool.')



def update_history(dts_pool, triggered_dt_idxs):

    suspended_mons = dts_pool[(dts_pool.active==True) & (dts_pool.suspended==True) ].dt_number.values
    dt_idxs = np.concatenate((triggered_dt_idxs,suspended_mons))
    for dt_idx in dts_pool[dts_pool.active==True].dt_number.values:
        if dt_idx in dt_idxs:
            dts_pool.loc[dts_pool.dt_number==dt_idx, 'history'] = [dts_pool.loc[dts_pool.dt_number==dt_idx, 'history'].values[0] + [1]]
        else:
            dts_pool.loc[dts_pool.dt_number==dt_idx, 'history'] = [dts_pool.loc[dts_pool.dt_number==dt_idx, 'history'].values[0] + [0]]
    handle_dts_redundancy(dts_pool, dt_idxs)
    
    
def handle_dts_redundancy(dts_pool, dt_idxs):    
    active_triggered_dt_idxs = []
    
#     for dt_idx in dt_idxs:
#         if dts_pool.loc[dts_pool.dt_number==dt_idx, 'active'].iloc[0]:
#             active_triggered_dt_idxs += [dt_idx]
    
    active_triggered_dt_idxs = list(dts_pool.loc[dts_pool.active==True].dt_number.unique())

    if len(active_triggered_dt_idxs) > 1:
        jaccard_map = {}
        dts_idxs_to_remove = []
        for dt_idx in active_triggered_dt_idxs:
            max_significance = -np.inf
            for other_dt_idx in active_triggered_dt_idxs:
                if other_dt_idx  == dt_idx:
                    pass
                dt_hist = dts_pool.loc[dts_pool.dt_number==dt_idx, 'history'].values[0]
                other_dt_hist = dts_pool.loc[dts_pool.dt_number==other_dt_idx, 'history'].values[0]
                
#                 print('dt_hist', dt_hist, 'other dt hist', other_dt_hist)
                
                size = len(dt_hist) if len(dt_hist) <= len(other_dt_hist) else len(other_dt_hist)

                map_key = str(dt_idx) + "_" + str(other_dt_idx) if dt_idx <= other_dt_idx else str(other_dt_idx) + "_" + str(dt_idx)

                if map_key in jaccard_map:
                    sim,p_val = jaccard_map[map_key]
                else:
                    sim,p_val = jaccard_test(dt_hist[-size:], other_dt_hist[-size:])
                    jaccard_map[map_key] = (sim,p_val)
                if max_significance < p_val:
                    max_significance = p_val
            if max_significance > .95:
                dts_idxs_to_remove += [dt_idx]

                
        if len(dts_idxs_to_remove) > 1:
            indipendent_dts_idxs_to_remove = []
            for dt_idx in dts_idxs_to_remove:
                dt_deps =  get_all_dt_deps(dt_idx, dts_pool)
                if not any(dt in dt_deps for dt in dts_idxs_to_remove):
                    indipendent_dts_idxs_to_remove += [dt_idx]
                    
            if len(indipendent_dts_idxs_to_remove) == 1: 
                dt_idx_to_keep = indipendent_dts_idxs_to_remove[0]
            if len(indipendent_dts_idxs_to_remove) > 1: 
                # find dt with highest score from the ones to remove
                dts_toremove_data = dts_pool[dts_pool.dt_number.isin(indipendent_dts_idxs_to_remove)]
                try:
                    dt_idx_to_keep = dts_toremove_data[dts_toremove_data.score == dts_toremove_data.score.min()].iloc[0]['dt_number']
                except:
                    print('error in finding dt to keep:', dts_toremove_data, 'row \\w max score:', dts_toremove_data[dts_toremove_data.score == dts_toremove_data.score.min()].iloc[0]['dt_number'], 'idx to remove:', indipendent_dts_idxs_to_remove)
                    dt_idx_to_keep = dts_idxs_to_remove
            else: ## 0 indipendent dt to remove
                # find dt with highest score from the ones to remove
                dts_toremove_data = dts_pool[dts_pool.dt_number.isin(dts_idxs_to_remove)]
                try:
                    dt_idx_to_keep = dts_toremove_data[dts_toremove_data.score == dts_toremove_data.score.min()].iloc[0]['dt_number']
                except:
                    print('error in finding dt to keep:', dts_toremove_data, 'row \\w max score:', dts_toremove_data[dts_toremove_data.score == dts_toremove_data.score.min()].iloc[0]['dt_number'], 'idx to remove:', dts_idxs_to_remove)
                    dt_idx_to_keep = dts_idxs_to_remove

            dts_idxs_to_remove = sorted(set(dts_idxs_to_remove)-set([dt_idx_to_keep]))

        #for dt_idx in dts_idxs_to_remove:
        #    print('Deactivating rule for redundancy', dt_idx)
        #    #deactivate_dt(dts_pool, dt_idx)


def get_all_dt_deps(dt_idx, dts_pool):
    deps =  list(dts_pool.loc[dts_pool.dt_number==dt_idx, 'dependency'].values[0])
    acc_deps = deps.copy()
    for dep in acc_deps:
        deps = list(set(deps + get_all_dt_deps(dep, dts_pool)))
    return deps




def get_new_dt_dep_level(dt_idxs, dts_pool):
    if len(dt_idxs) == 0:
        return 1
    else: 
        return max([dts_pool[dts_pool.dt_number==dt_idx].dep_level.values[0] for dt_idx in dt_idxs])+1
    
    
def dt_far_score(y,y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return fp/(fp+tn) if (fp+tn) > 0. else 0.



def online_monitoring(mon, trace, t, sensor_columns, verbose=False):
    
    try:
        rob = mon.update(t, [(c, trace.iloc[t][c]) for c in sensor_columns])
    except RTAMTException as e:
        print('RTAMTException:', e, '\nFormula:', mon.spec,  'trace length:', len(trace), 't:', t, 'trace at t:', trace.iloc[t])
        return False
    
    if verbose:

        for c in sensor_columns:
            plt.plot(trace.iloc[0:t+1][c], label=c)
        plt.legend()
        plt.show()

        print('Monitoring STL formula:', mon.spec, ' | Max robustness:', rob)

    return True if rob >= 0 else False


def online_monitoring_rob(mon, trace, t, sensor_columns, verbose=False):
    
    try:
        rob = mon.update(t, [(c, trace.iloc[t][c]) for c in sensor_columns])
    except RTAMTException as e:
        print('RTAMTException:', e, '\nFormula:', mon.spec,  'trace length:', len(trace), 't:', t, 'trace at t:', trace.iloc[t])
        return False
    
    if verbose:

        for c in sensor_columns:
            plt.plot(trace.iloc[0:t+1][c], label=c)
        plt.legend()
        plt.show()

        print('Monitoring STL formula:', mon.spec, ' | Max robustness:', rob)

    return rob


def get_monitor(formula, sensor_columns, column_types):
  
    spec = rtamt.StlDiscreteTimeOnlineSpecification()

    formula = re.sub(r'(\[[0-9:]*)\.([0-9]+)',  r'\1', formula)
    
    for c in sensor_columns:
        spec.declare_var(c, column_types[c])

    spec.spec = formula

    try:
        spec.parse()

        h = StlHorizon()
        for s in spec.ast.specs:
            horizon = h.visit(s, None)
            
        spec.pastify()

    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        return None

    #spec.h = float(horizon)
    assert float(horizon).is_integer()
    #spec.h = math.ceil(spec.h) if spec.h > 1. else 1
    spec.h = int(horizon)
    
    return spec


def get_offline_monitor(formula, sensor_columns, column_types):
  
    spec = rtamt.StlDiscreteTimeOfflineSpecification()

    formula = re.sub(r'(\[[0-9:]*)\.([0-9]+)',  r'\1', formula)
    
    for c in sensor_columns:
        spec.declare_var(c, column_types[c])

    spec.spec = formula

    try:
        spec.parse()

        h = StlHorizon()
        for s in spec.ast.specs:
            horizon = h.visit(s, None)
            

    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        return None

    assert float(horizon).is_integer()
    spec.h = int(horizon)
    
    return spec


def reset_all_monitors(dts_pool):
    dts_pool.loc[dts_pool.active==True, 'counter'] = 0  
    dts_pool.loc[dts_pool.active==True, 'suspended'] = False

    for dt_idx in dts_pool[(dts_pool.active==True) & (dts_pool.suspended==False) ].dt_number.values:
        dt = dts_pool[dts_pool.dt_number==dt_idx].iloc[0]
        mon = dt['mon']
        dts_pool.loc[dts_pool.dt_number==dt_idx, 'mon'] = get_monitor(mon.spec)

def reset_all_monitors_offline(dts_pool):
    dts_pool.loc[dts_pool.active==True, 'counter'] = 0  
    dts_pool.loc[dts_pool.active==True, 'suspended'] = False

def advance_all_monitors(dts_pool):
    dts_pool.loc[(dts_pool.active==True) & (dts_pool.suspended==False) , 'counter'] += 1 

def suspend_monitors(dts_pool, triggered_dt_idxs):
    dts_pool.loc[dts_pool.dt_number.isin(triggered_dt_idxs), 'suspended'] = True 


    

def test_formula_offline_extended(arg):
    spec, trace_data = arg
    if len(trace_data['time']) <= spec.h:
        print("Trace shorther than Horizon")
        return -1,-1, -1, spec.h, np.array([-1])
        
    rob=np.array([r[1] for r in evaluate_spec(spec, trace_data)])

    assert len(rob) == len(trace_data['time']), str(len(rob)) + " " + str(len(trace_data['time'])) + " " + str(spec.h)
    
    #if spec.h > 1:
    if spec.h >= 1:
        try:
            rob = rob[ : -int(spec.h) ]
        except Exception as e:
            print(e, spec.h)
            assert False
    
    t_first = np.argmax(rob >= 0)
    rob_first = rob[t_first]
    
    max_rob = np.nanmax(rob)
    return t_first, rob_first, max_rob, spec.h, rob

# function that acts as wrapper for monitoring call, as rtamt does not work with traces of len 1
#to overcome this, we add a fake temporal instant in such case (this is automatically ignore by rtmat, which will return a rob vector of len 1)
def evaluate_spec(spec, trace_data):
    trace_data_temp = trace_data.copy()
    if len(trace_data_temp['time']) == 1:
        trace_data_temp['time'] = [0,0]
    return spec.evaluate(trace_data_temp)
    
def test_formula_online(mon, trace, sensor_columns):
    for t in range(len(trace)):
        sat = online_monitoring(mon, trace, t, sensor_columns)
        if t >= mon.h and sat:
            return t
    return -1

def init_all_active_monitors(dts_pool, sensor_columns, column_types):
    for i in dts_pool[dts_pool.active==True].dt_number.values:
        dt = dts_pool[dts_pool.dt_number==i].iloc[0]
        dts_pool.iloc[i, dts_pool.columns.get_loc('mon')] = get_monitor(json.loads(dt['formula']))

def init_all_active_offline_monitors(dts_pool, sensor_columns, column_types):
    for i in dts_pool[dts_pool.active==True].dt_number.values:
        dt = dts_pool[dts_pool.dt_number==i].iloc[0]
        dts_pool.iloc[i, dts_pool.columns.get_loc('mon')] = get_offline_monitor(json.loads(dt['formula']), sensor_columns, column_types)
    


####################################################################################################
#WARMUP: train the monitor
####################################################################################################


def train_process(args):
    
    k = args[0] 
    seed = args[1]
    # training traces
    data = args[2]
    # max and ratio of traces to be consdiered as good traces (work in both modes)
    max_good_witnesses = args[3]
    good_witnesses_ratio = args[4] 
    # params to allow mulitple iteration of the framework over the same trainset
    training_set_repetitions = args[5]
    early_stopping = args[6]
    patience = args[7]
    # technical stuff
    save_folder = args[8]
    dataset_name = args[9]
    output_pickle_file_suffix = args[10]
    # BOH
    es_epsilon = args[11]
    # number of time to which augment each training trace
    n_augs = args[12]
    # technical stuff
    sensor_columns = args[13]
    gp_instance = args[14]
    # hyperparams of the EA
    max_gen = args[15]
    pop_size = args[16]
    cxpb = args[17]
    mutpb = args[18]
    # min score (FAR or MCC) that a fomrula shall have to be returned from the EA (note that automatically this is for FAR max_score_ea, while for MCC 1-max_score_ea)
    max_score_ea = args[19]
    # min accuracy that a fomrula shall have to be returned from the EA
    min_accuracy = args[20]
    # technical stuff
    column_types = args[21]
    # EMA factor for discarging formula from the pool
    score_beta = args[22]
    # value that single formula shall be lower in order to be no deactivated
    max_score_pool = args[23]
    # traces to consider for evaluation purposes
    val_traces = args[24]
    # wehther to returns a single solution of the EA or all those in the front
    return_multiple_formulas_ea = args[25]
    # noise for data augmentation, which applies to plain traces as well as bad and good ones
    noise_scale = args[26]
    noise_kind = args[27]
    # when good and bad traces used in framework, whether to keep them or remove from those cosndiered for training
    remove_bad_good_from_training = args[28]
    # how many agumented traces to get for each bad and good trace (work in both modes)
    num_augment_bad_good = args[29]
    # max and ratio of traces to be consdiered as bad traces (work in both modes)
    max_bad_witnesses = args[30]
    bad_witnesses_ratio = args[31]

    col_map_std = args[32]

    # get the number of good and failure traces in the training set
    tot_good = len(data[data.failure == 0].serial_number.unique())
    tot_failure = len(data[data.failure == 1].serial_number.unique())

    #  max_good_witnesses == -1 implies that no max_good_witnesses
    if max_good_witnesses == -1:
        max_good_witnesses = tot_good
    else:
        max_good_witnesses = min(tot_good, max_good_witnesses)    

    #  max_bad_witnesses == -1 implies that no max_bad_witnesses
    if max_bad_witnesses == -1:
        max_bad_witnesses = tot_failure
    else:
        max_bad_witnesses = min(tot_failure, max_bad_witnesses)

    # given max and ratio values, determine how many good and bad witnesses to consider at most
    num_good_witnesses = int(min(max_good_witnesses, round(len(data[data.failure == 0].serial_number.unique().tolist())*good_witnesses_ratio))) 
    num_bad_witnesses =  int(min(max_bad_witnesses, round(len(data[data.failure == 1].serial_number.unique().tolist())*bad_witnesses_ratio)))

    # if we want to use good and bad traces within the EA, prepare the data and aass it to the EA object
    if GP_STL.goodbad_traces_at_ea_generation > 0:
        print("Generating good and bad traces to use within the EA")

        def help_augmented_traces(w):
            return get_augmented_traces(data, w, len(data[data['serial_number'] == w]), seed, num_augment_bad_good, sensor_columns, noise_scale, noise_kind, col_map_std)
        good_witnesses_ea_data = []
        bad_witnesses_ea_data = []

        p=ProcessPool(GP_STL.processes)
        p.restart()
        good_witnesses_ea_data = p.map(help_augmented_traces, np.unique(data[data["failure"] == 0]['serial_number'].values))
        p.close()
        p.join()        
        p=ProcessPool(GP_STL.processes)
        p.restart()
        bad_witnesses_ea_data = p.map(help_augmented_traces, np.unique(data[data["failure"] == 1]['serial_number'].values))
        p.close()
        p.join()        
        good_witnesses_ea_data = [x for y in good_witnesses_ea_data for x in y]
        bad_witnesses_ea_data = [x for y in bad_witnesses_ea_data for x in y]
        GP_STL.num_good_witnesses = num_good_witnesses
        GP_STL.num_bad_witnesses = num_bad_witnesses
        # set witnesses to zero outside the EA to skip nedless calculations
        num_good_witnesses = 0
        num_bad_witnesses = 0

    
    random.seed(seed) # for random traces concatenation
    np.random.seed(seed) # for generating the seeds of the genetic programming algorithm 
    
    fw_stats = pd.DataFrame([], columns=['iteration', 'unit', 't_stat', 'p_stat', 'is_good', 'n_p', 'nn', 'tp', 'fp', 'train','nprop'])
    fw_stats_val = pd.DataFrame([], columns=['iteration', 'tp', 'fp', 'tn', 'fn'])
    fw_stats_full = pd.DataFrame([], columns=['iteration', 'tp', 'fp', 'tn', 'fn'])
    dts_pool = pd.DataFrame([], columns=['dt_number', 'human_readable_formula', 'iteration', 'unit', 'dt_p', 'dt_far', 'dt_tp', 'dt_fp', 
                                         'dt_nn','dt_np', 'dependency', 'train', 'score', 'active', 'formula', 'dep_level', 'dt', 'history', 'mon', 'suspended', 'counter', 'accuracy', 'scl_ea'])
    
    n_p=0 # number of positive ground truth traces (failure traces)
    nn=0 # number of negative ground truth traces (good traces)
    tp=0 # true positive counter
    fp=0 # false positive counter
    
    start_train = time.time()

    print("Training on", len(data.serial_number.unique()), "traces, repeated at most", training_set_repetitions, "times.")
    print("Using early stopping on F1:", early_stopping)

    # prepare witnesses to use outside the EA. If use within the EA these result in empty lists, as the corresponing num for sampling are zero
    good_witnesses_ea = random.sample(data[data.failure == 0].serial_number.unique().tolist(), num_good_witnesses)
    print("Number of good witnesses (outside ea)", len(good_witnesses_ea), num_good_witnesses)
    bad_witnesses_ea = random.sample(data[data.failure == 1].serial_number.unique().tolist(), num_bad_witnesses)
    print("Number of bad witnesses (outside ea)", len(bad_witnesses_ea), num_bad_witnesses)

    # determine train units, removing those used as witnesses outside the EA if specified by remove_bad_good_from_training
    train_units = []
    for i in range(training_set_repetitions):
        if remove_bad_good_from_training:
            train_units.extend(list(get_shuffled_units_warmup(data[~data.serial_number.isin(good_witnesses_ea+bad_witnesses_ea)], seed+i, start_with_failure=True)))
        else:
            train_units.extend(list(get_shuffled_units_warmup(data, seed+i, start_with_failure=True)))

    def eval_all_val(val_unit):
        val_trace = get_unit_trace(val_traces, val_unit, discretized=False)
        eval_formula_outcome = False
        for j in dts_pool[(dts_pool.active==True) & (dts_pool.suspended==False)].dt_number.values:
            dt = dts_pool[dts_pool.dt_number==j].iloc[0]
            mon = dt['mon']#.tolist()[0])
            scl_ea = dt['scl_ea']
            temp_trace = val_trace.copy()
            temp_trace[sensor_columns] = scl_ea.transform(temp_trace[sensor_columns])
            trace_data = {}
            trace_data['time'] = range(len(temp_trace)) #if len(temp_trace)>1 else [0,0]
            trace_data.update({c:temp_trace[c].values.copy() for c in sensor_columns})
            eval_formula_outcome = np.max(test_formula_offline_extended((mon, trace_data))[-1]) >= 0 
            if eval_formula_outcome:
                return (eval_formula_outcome, bool(val_trace['failure'].values[0]))
        return (eval_formula_outcome, bool(val_trace['failure'].values[0]))
            
    def eval_all_full(full_unit):
        full_trace = get_unit_trace(data, full_unit, discretized=False)
        eval_formula_outcome = False
        for j in dts_pool[(dts_pool.active==True) & (dts_pool.suspended==False)].dt_number.values:
            dt = dts_pool[dts_pool.dt_number==j].iloc[0]
            mon = dt['mon']#.tolist()[0])
            scl_ea = dt['scl_ea']
            temp_trace = full_trace.copy()
            temp_trace[sensor_columns] = scl_ea.transform(temp_trace[sensor_columns])
            trace_data = {}
            trace_data['time'] = range(len(temp_trace)) #if len(temp_trace)>1 else [0,0]
            trace_data.update({c:temp_trace[c].values.copy() for c in sensor_columns})
            eval_formula_outcome = np.max(test_formula_offline_extended((mon, trace_data))[-1]) >= 0 
            if eval_formula_outcome:
                return (eval_formula_outcome, bool(full_trace['failure'].values[0]))
        return (eval_formula_outcome, bool(full_trace['failure'].values[0]))
    
    best_score = 0. # Highest F1 score witnessed so far by the framework
    counter = 0  # Counter for how many iterations the score has not improved
    for i,unit in enumerate(train_units):

        t_stat=-1
        p_stat=-1
        
        gc.collect()

        trace=get_unit_trace(data, unit, discretized=False)
        
        is_good = not trace.iloc[-1].failure    
        
        if is_good:
            nn += 1
            # ensure that good witnesses list outside the EA changes
            if unit not in good_witnesses_ea and len(data[data.serial_number == unit]) >= 2 and num_good_witnesses > 0:
                if len(good_witnesses_ea) < num_good_witnesses: 
                    good_witnesses_ea.append(unit)
                else:
                    good_witnesses_ea[random.randint(0, num_good_witnesses-1)] = unit
        else: 
            n_p += 1     
            # ensure that bad witnesses list outside the EA changes
            if unit not in bad_witnesses_ea and len(data[data.serial_number == unit]) >= 2 and num_bad_witnesses > 0:
                if len(bad_witnesses_ea) < num_bad_witnesses: 
                    bad_witnesses_ea.append(unit)
                else:
                    bad_witnesses_ea[random.randint(0, num_bad_witnesses-1)] = unit

        # traces used by the genetic algorithm
        traces_data=[]
                
        print('>>> Trace:', i+1, 'unit:', unit, ' is_good:', is_good)

        
        # Iterate through traces until the end ( = failure) is reached or a failure is predicted
        
        failure_triggered = False

        reset_all_monitors_offline(dts_pool)


        # auxiliary code to test all monitors on validation traces (val_traces). For efficeincy this is done every 5% of training length
        if i % int(round(len(train_units)*0.05)) == 0:
            val_res = []
            val_gt = []

            p=ProcessPool(GP_STL.processes)
            p.restart()
            res_parallel = np.asarray(p.map(eval_all_val, np.unique(val_traces["serial_number"].values.tolist())))
            p.close()
            p.join()
            val_gt = res_parallel[:, 1]
            val_res = res_parallel[:, 0]
            
            tn_val, fp_val, fn_val, tp_val = confusion_matrix(val_gt, val_res).ravel()      





            
        # auxiliary code to test all monitors on all training traces (data). For efficeincy this is done every 5% of training length
        if i % int(round(len(train_units)*0.05)) == 0:
            full_res = []
            full_gt = []

            p=ProcessPool(GP_STL.processes)
            p.restart()
            res_parallel = np.asarray(p.map(eval_all_full, np.unique(data["serial_number"].values.tolist())))
            p.close()
            p.join() 
            full_gt = res_parallel[:, 1]
            full_res = res_parallel[:, 0]
            
            tn_full, fp_full, fn_full, tp_full = confusion_matrix(full_gt, full_res).ravel()  

        # dict used to store the result of the execution of all active offline monitors on the current trances
        offline_mon_comput = {}

        # execute all active offline monitors on the current trace
        for j in dts_pool[(dts_pool.active==True) & (dts_pool.suspended==False)].dt_number.values:
            dt = dts_pool[dts_pool.dt_number==j].iloc[0]
            mon = dt['mon']#.tolist()[0])
            scl_ea = dt['scl_ea']
            temp_trace = trace.copy()
            temp_trace[sensor_columns] = scl_ea.transform(temp_trace[sensor_columns])
            trace_data = {}
            trace_data['time'] = range(len(temp_trace)) #if len(temp_trace)>1 else [0,0]
            trace_data.update({c:temp_trace[c].values.copy() for c in sensor_columns})
            offline_mon_comput[j] = test_formula_offline_extended((mon, trace_data))[-1] >= 0          
            
        for t in range(len(trace)):
            
            advance_all_monitors(dts_pool)

            triggered_dt_idxs=[]
            
            # check whether you have something to monitor (active, i.e., not deleted, and not suspended)
            if len(dts_pool[(dts_pool.active==True) & (dts_pool.suspended==False ) ]) > 0:
                
                max_h = 0 

                for j in dts_pool[(dts_pool.active==True) & (dts_pool.suspended==False)].dt_number.values: # skip unactive or suspended formulas
                    dt = dts_pool[dts_pool.dt_number==j].iloc[0]
                    mon = dt['mon']#.tolist()[0])

                    
                    if dt['counter'] <= len(offline_mon_comput[j]): # skip still non monitorable formulas, i.e., those with counter < horizon       
                        
                        sat = offline_mon_comput[j][t]

                        if sat:
            
                            if not failure_triggered:
                                failure_triggered=True
                                if not is_good: # teacher forcing
                                    t_stat = t
                                    p_stat = len(trace) - t - 1 
                                    
                            if mon.h > max_h:
                                max_h = int(mon.h)
                                #triggered_dt_idx = j # dunno if needed 
                            triggered_dt_idxs += [j]        

 
            if failure_triggered and not is_good: # TP, execute at most once per trace
                if t >= 2: # we need at least two time points to split traces in good and bad to perform a new property extraction
                    traces_data = get_augmented_traces(data, unit, t-1, seed, n_augs, sensor_columns, noise_scale, noise_kind, col_map_std) # N.B:get_augmented_traces deve pescare da 0 a t-max_h (incluso)
                tmp_prova = triggered_dt_idxs.copy()
                # update statistics of all triggered formulas
                triggered_dt_idxs = update_dt_data(dts_pool, triggered_dt_idxs, t, trace, is_good, score_beta, max_score_pool)
                print("sono uguali?", triggered_dt_idxs, tmp_prova)
                if len(triggered_dt_idxs) > 0:
                    tp += 1
                else:
                    assert False, "This should not have happend"
                # note that with this break we are not scanning all prefixes of a trace, as we stop the first prefix where (a set of) monitor triggers 
                break
                
            elif failure_triggered and is_good:# and not missed: # FP    
                # suspend triggered properties
                suspend_monitors(dts_pool, triggered_dt_idxs)
                update_dt_data(dts_pool, triggered_dt_idxs, t, trace, is_good, score_beta, max_score_pool) 

            
            if t == len(trace) - 1:
                # FP or #FN or TN

                if failure_triggered and is_good: # FP
                    fp += 1
                    t_stat = -1
                    p_stat = -1
                    traces_data = []
                    
                elif failure_triggered and not is_good: # TP
                    assert False, "Impossibile, uscito con il break"
                    
                elif not failure_triggered and is_good: # TN
                    print("La traccia è un vero negativo.")
                    
                elif not failure_triggered and not is_good: # FN
                    t_stat=t
                    p_stat=0
                    max_h = 0
                    if t>=2:
                        traces_data = get_augmented_traces(data, unit, t, seed, n_augs, sensor_columns, noise_scale, noise_kind, col_map_std)

                update_dt_miss_data(dts_pool, is_good, score_beta, max_score_pool)

        if not is_good:
            print('Failures t:', t_stat)
            print('Failure prediction preemptiveness :', p_stat)
        print('Detection:',  failure_triggered)
                
        update_history(dts_pool, triggered_dt_idxs)
            
        # add unit stats
        fw_stats.loc[len(fw_stats), :] = [i, unit, t_stat, p_stat, is_good, n_p, nn, tp, fp, True, len(dts_pool[dts_pool.active==True])] # last means train = True
        fw_stats_val.loc[len(fw_stats_val), :] = [i, tp_val, fp_val, tn_val, fn_val] # last means train = True       
        fw_stats_full.loc[len(fw_stats_full), :] = [i, tp_full, fp_full, tn_full, fn_full] # last means train = True       

        
        # triggered_dt_idxs will be not empty iff TP and still active after redundancy removal (executed in update_history)
        triggered_dt_idxs = sorted(dts_pool[(dts_pool.active==True) & (dts_pool.dt_number.isin(triggered_dt_idxs))].dt_number.values)

        # search for new formulas
        
        if len(traces_data) > 0 :
                
            print('Properties extraction...')
            extraction_start = time.time()

            # construct the set of good and bad traces to use if we are not in the mode where everyting is done random within the EA (in such a case the two lists contains all bad/good training data)
            if not GP_STL.goodbad_traces_at_ea_generation > 0:
                good_witnesses_ea_data = []
                for gw in good_witnesses_ea:
                    #good_witnesses_ea_data.append(data[data['serial_number'] == gw].copy())
                    good_witnesses_ea_data.extend(get_augmented_traces(data, gw, len(data[data['serial_number'] == gw]), seed, num_augment_bad_good, sensor_columns, noise_scale, noise_kind, col_map_std))
    
                bad_witnesses_ea_data = []
                for bw in bad_witnesses_ea:
                    #bad_witnesses_ea_data.append(data[data['serial_number'] == bw].copy())
                    bad_witnesses_ea_data.extend(get_augmented_traces(data, bw, len(data[data['serial_number'] == bw]), seed, num_augment_bad_good, sensor_columns, noise_scale, noise_kind, col_map_std))

                
            formulas, windows, accuracies, far_gps, horizons, formulas_noNorm, scl_ea = gp_instance.run_gp_stl(copy.deepcopy(traces_data), max_gen=max_gen, pop_size=pop_size, cxpb=cxpb, mutpb=mutpb, verbose=False, 
                                                               alg_seed=seed+int.from_bytes(str(unit).encode('utf-8'), "little")%1000000, 
                                                               train_good_traces=[], good_witnesses_ea=good_witnesses_ea_data, bad_witnesses_ea=bad_witnesses_ea_data, maxfar = max_score_ea, minacc = min_accuracy,
                                                               return_multiple_formulas_ea=return_multiple_formulas_ea)
            print('Properties extraction performed in ', time.time() - extraction_start, 'secs:')
            
            if len(formulas) == 0:
                print('no_formulas extracted')
            else:
                print("Formulas extracted:", len(formulas))
                for fidx in range(len(formulas)):

                    formula_denorm = formulas[fidx] # formula where constanst have been de-normalized, to kae the formula fully interpretable DO NOT USE (NUMERICAL ISSUES)
                    window = windows[fidx]
                    accuracy = accuracies[fidx]
                    far_gp = far_gps[fidx]
                    horizon_ea = horizons[fidx]
                    formula = formulas_noNorm[fidx] # formula with normalized constanst. Before applying use the correspoding scaler
                    
                    print(formula_denorm, '(w:'+ str(window) +')', 'acc', accuracy, 'far', far_gp, 'h', horizon_ea, '\n-----------------------------------------------------')

                    temp_trace = traces_data[0].copy()
                    temp_trace[sensor_columns] = scl_ea.transform(temp_trace[sensor_columns])
                    
                    mon = get_monitor(formula, sensor_columns, column_types)
                    ext_t_online = test_formula_online(mon, temp_trace, sensor_columns)
    
                    offline_mon = get_offline_monitor(formula, sensor_columns, column_types)
                    trace_data = {}
                    trace_data['time'] = range(len(temp_trace)) #if len(temp_trace)>1 else [0,0]
                    trace_data.update({c:temp_trace[c].values.copy() for c in sensor_columns})
                    ext_t_offline, _, _, horizon_offline, _ = test_formula_offline_extended((offline_mon, trace_data))
                    
                    if ext_t_offline < 0:
                        print('Extracted property with accuracy:', accuracy, 'and far:', far_gp, 'doesn\'t satisfy the original trace.')#' This happened', nonsat_nscazz, 'times') 
                    else:
                        print('Extracted property with accuracy:', accuracy, 'and far:', far_gp,'satisfies the original trace at ONLINE', ext_t_online, ', OFFLINE', 
                              ext_t_offline, '(h:', horizon_offline, ') the genetic detection point was', len(traces_data[0]) - window , '(h:', horizon_ea, ") trace len ", len(traces_data[0]))
                        
                    dts_pool.loc[len(dts_pool), : ] = [len(dts_pool), json.dumps(formula_denorm), i, unit, p_stat+max_h, -1, 0., 0., 0., 0., 
                                                       triggered_dt_idxs, True, 0., True, json.dumps(formula), 
                                                       get_new_dt_dep_level(triggered_dt_idxs, dts_pool), None, [1], 
                                                       get_offline_monitor(formula, sensor_columns, column_types), False, 0, accuracy, scl_ea]
                        
                    print('Accuracy > ', min_accuracy, '(', accuracy, ') and FAR <=', max_score_ea, '(', far_gp, '), formula added to db:\n', formula_denorm)

        
        elif len(triggered_dt_idxs) > 0:
            print("No properties extraction due to short traces")
            
        # update eval smetrics
        precision = tp/(tp+fp) if (tp+fp) > 0 else 1.
        far = fp/nn if nn > 0 else 0.
        recall = tp/n_p  if n_p > 0 else 1.
        f1_score = (2*precision*recall)/(precision+recall) if precision*recall > 0 else 0.

        
        print("Running stats:", 'precision', precision, 'far', far, 'recall', recall, 'f1_score', f1_score, 'active formulas', len(dts_pool[(dts_pool.active==True) & (dts_pool.suspended==False)]))

        precision_val = tp_val/(tp_val+fp_val) if (tp_val+fp_val) > 0 else 1.
        far_val = fp_val/(tn_val + fp_val) if tn_val + fp_val > 0 else 0.
        recall_val = tp_val/(tp_val + fn_val)  if tp_val + fn_val > 0 else 1.
        f1_score_val = (2*precision_val*recall_val)/(precision_val+recall_val) if precision_val*recall_val > 0 else 0.


        precision_full = tp_full/(tp_full+fp_full) if (tp_full+fp_full) > 0 else 1.
        far_full = fp_full/(tn_full + fp_full) if tn_full + fp_full > 0 else 0.
        recall_full = tp_full/(tp_full + fn_full)  if tp_full + fn_full > 0 else 1.
        f1_score_full = (2*precision_full*recall_full)/(precision_full+recall_full) if precision_full*recall_full > 0 else 0.
        
        print("Running stats VAL:", 'precision', precision_val, 'far', far_val, 'recall', recall_val, 'f1_score', f1_score_val)
        print("Running stats FULL:", 'precision', precision_full, 'far', far_full, 'recall', recall_full, 'f1_score', f1_score_full)
        
        res = {'fw_stats':fw_stats, 'dts_pool':dts_pool.drop(columns='mon'), 'precision':precision, 'far':far, 'recall':recall, 'f1_score':f1_score,
              'fw_stats_val':fw_stats_val, 'precision_val':precision_val, 'far_val':far_val, 'recall_val':recall_val, 'f1_score':f1_score_val,
              'fw_stats_full':fw_stats_full, 'precision_full':precision_full, 'far_full':far_full,'recall_full':recall_full, 'f1_score_full':f1_score_full}    


        write_pickle(res, save_folder + 'train_stl_'+ str(seed) +'_results_'+ dataset_name+output_pickle_file_suffix +'.pickle')


        # Check if the new score is better than the best score seen so far
        if f1_score > best_score + es_epsilon:
            print("New best f1_score:", f1_score)
            best_score = f1_score
            counter = 0  # Reset the counter as we have seen an improvement
        else:
            counter += 1  # Increment the counter as there was no improvement

        # If the counter reaches the patience limit, stop the process
        if counter >= patience and early_stopping:
            print("Stopping early after", i+1, "framework iterations.")
            break

           
                
    print('Train time:', time.time() - start_train)        
    
    # update eval smetrics
    precision = tp/(tp+fp) if (tp+fp) > 0 else 1.
    far = fp/nn if nn > 0 else 0.
    recall = tp/n_p  if n_p > 0 else 1.
    f1_score = (2*precision*recall)/(precision+recall) if precision*recall > 0 else 0.

    precision_val = tp_val/(tp_val+fp_val) if (tp_val+fp_val) > 0 else 1.
    far_val = fp_val/(tn_val + fp_val) if tn_val + fp_val > 0 else 0.
    recall_val = tp_val/(tp_val + fn_val)  if tp_val + fn_val > 0 else 1.
    f1_score_val = (2*precision_val*recall_val)/(precision_val+recall_val) if precision_val*recall_val > 0 else 0.


    precision_full = tp_full/(tp_full+fp_full) if (tp_full+fp_full) > 0 else 1.
    far_full = fp_full/(tn_full + fp_full) if tn_full + fp_full > 0 else 0.
    recall_full = tp_full/(tp_full + fn_full)  if tp_full + fn_full > 0 else 1.
    f1_score_full = (2*precision_full*recall_full)/(precision_full+recall_full) if precision_full*recall_full > 0 else 0.
    
        
    res = {'fw_stats':fw_stats, 'dts_pool':dts_pool.drop(columns='mon'), 'precision':precision, 'far':far, 'recall':recall, 'f1_score':f1_score,
          'fw_stats_val':fw_stats_val, 'precision_val':precision_val, 'far_val':far_val, 'recall_val':recall_val, 'f1_score_val':f1_score_val,
              'fw_stats_full':fw_stats_full, 'precision_full':precision_full, 'far_full':far_full, 'recall_full':recall_full, 'f1_score_full':f1_score_full}      
    
    return res

	
####################################################################################################
#ONLINE: test the monitor
####################################################################################################



def test_process(i_seed, seed,sensor_columns,column_types, test_simulated_online=False, test_data=None, data=None, train_res=None):

    random.seed(seed) # for random traces concatenation
    np.random.seed(seed) # for generating the seeds of the genetic programming algorithm 
    
    fw_stats = train_res[i_seed]['fw_stats']
    dts_pool_whole = train_res[i_seed]['dts_pool']


    dts_pool_whole['mon']= None
    
    dts_pool_whole=dts_pool_whole.astype({"dependency": object, "history": object})
    
    
    init_all_active_offline_monitors(dts_pool_whole,sensor_columns,column_types)

    print(dts_pool_whole)

    df = pd.DataFrame(columns=['precision_val', 'far_val', 'recall_val', 'f1_score_val', 'mcc_score_val', 'preempt_allfailure', 'preempt_failureandtriggered'])
    for i in tqdm(range(len(dts_pool_whole))):
        dts_pool = dts_pool_whole.iloc[:i+1].copy()

        print("Test on", len(test_data.serial_number.unique()), "traces.")
        
        def eval_all_val(val_unit):
            val_trace = get_unit_trace(test_data, val_unit, discretized=False)
            eval_formula_outcome_return = False
            preempt = 0
            for j in dts_pool[(dts_pool.active==True) & (dts_pool.suspended==False)].dt_number.values:
                dt = dts_pool[dts_pool.dt_number==j].iloc[0]
                mon = dt['mon']#.tolist()[0])
                scl_ea = dt['scl_ea']
                temp_trace = val_trace.copy()
                temp_trace[sensor_columns] = scl_ea.transform(temp_trace[sensor_columns])
                trace_data = {}
                trace_data['time'] = range(len(temp_trace)) #if len(temp_trace)>1 else [0,0]
                trace_data.update({c:temp_trace[c].values.copy() for c in sensor_columns})
                t_first, _, _, _, eval_formula_outcome = test_formula_offline_extended((mon, trace_data))
                eval_formula_outcome = np.max(eval_formula_outcome) >= 0 
                if (len(val_trace)-1)-t_first >= preempt and eval_formula_outcome:
                    preempt = (len(val_trace) - 1) - t_first
                    eval_formula_outcome_return = True
            return (eval_formula_outcome_return, bool(val_trace['failure'].values[0]), preempt)
        # auxiliary code to test all monitors on validation traces (val_traces). For efficeincy this is done every 5% of training length
        val_res = []
        val_gt = []
        val_preempt = []

        p=ProcessPool(20)
        p.restart()
        res_parallel = np.asarray(p.map(eval_all_val, np.unique(test_data["serial_number"].values.tolist())))
        p.close()
        p.join()
        val_gt = res_parallel[:, 1]
        val_res = res_parallel[:, 0]
        val_preempt = res_parallel[:, 2]
        
        tn_val, fp_val, fn_val, tp_val = confusion_matrix(val_gt, val_res).ravel() 
        
        precision_val = tp_val/(tp_val+fp_val) if (tp_val+fp_val) > 0 else 1.
        far_val = fp_val/(tn_val + fp_val) if tn_val + fp_val > 0 else 0.
        recall_val = tp_val/(tp_val + fn_val)  if tp_val + fn_val > 0 else 1.
        f1_score_val = (2*precision_val*recall_val)/(precision_val+recall_val) if precision_val*recall_val > 0 else 0.
        mcc_score_val = ((tp_val * tn_val) - (fp_val * fn_val))/math.sqrt((tp_val + fp_val) * (tp_val + fn_val) * (tn_val + fp_val) * (tn_val + fn_val)) if math.sqrt((tp_val + fp_val) * (tp_val + fn_val) * (tn_val + fp_val) * (tn_val + fn_val)) > 0 else 0.
        preempt_allfailure = np.mean(val_preempt[val_gt==True])
        preempt_failureandtriggered = np.mean(np.asarray(val_preempt)[np.logical_and(val_gt==True, val_res==True)])

        data = {'precision_val': precision_val, 'far_val': far_val, 'recall_val': recall_val, 'f1_score_val': f1_score_val, 'mcc_score_val': mcc_score_val, 'preempt_allfailure': preempt_allfailure, 'preempt_failureandtriggered': preempt_failureandtriggered}
        df = df.append(data, ignore_index=True)


    print("TEST:", 'precision', precision_val, 'far', far_val, 'recall', recall_val, 'f1_score', f1_score_val, "MCC", mcc_score_val)

    return df    


    ## DETERMINE FILE TO USE AND SAVE
def main(params_dict, is_test):
	save_folder = './runs/'

	print(params_dict)

	if is_test:
		cur_timestamp = params_dict["cur_timestamp"]
	else:
		cur_timestamp = str(datetime.datetime.now()).replace(" ", "_")
		params_dict["cur_timestamp"] = cur_timestamp

	with open(save_folder + params_dict["dataset_name"] + "_" + params_dict["output_file_suffix"] + "_" + cur_timestamp + "_params", 'w') as file:
		file.write(str(params_dict))

	output_pickle_file_suffix = "_" + params_dict["output_file_suffix"]  + "_" + cur_timestamp

	dataset_name = params_dict["dataset_name"]

	if is_test:
		train_res_path = save_folder + 'train_stl_results_'+dataset_name + output_pickle_file_suffix +'.pickle'


	if dataset_name == "CMAPSS":
		train_output_file = "../dataset/CMAPSSData/trainset.csv" 
		test_output_file = "../dataset/CMAPSSData/testset.csv" 
		test_rul_file = "../dataset/CMAPSSData/RUL_FD001.txt" 
	elif dataset_name == "SMART_2016":
		train_output_file = "../dataset/smart/2016/trainset.csv" 
		test_output_file = "../dataset/smart/2016/testset.csv" 
	elif dataset_name == "SMART_full":
		train_output_file = "../dataset/smart/full/trainset.csv" 
		test_output_file = "../dataset/smart/full/testset.csv" 
	elif dataset_name == "TEP":
		train_output_file = "../dataset/TEP/trainset.csv" 
		test_output_file = "../dataset/TEP/testset.csv" 
	else:
		assert False, "Dataset not supported"

	num_aug_diff = 0
	num_aug_pctchange = 0


	## DATA PREPROCESSING


	original_training_data=pd.read_csv(train_output_file, sep=',', header=0, encoding="UTF-8")
	original_test_data=pd.read_csv(test_output_file, sep=',', header=0, encoding="UTF-8")


	if dataset_name == "CMAPSS":
		def add_string_in_columns(df, string):
			#print("BAU")
			df.columns = [col.replace("sensor", f"sensor{string}").replace("setting", f"setting{string}") for col in df.columns]
			return df

		test_rul = pd.read_csv(test_rul_file, header=None, names=["RUL"]).reset_index().rename(columns={"index": "serial_number"})
		test_rul["serial_number"] +=1
		test_rul["serial_number"] = test_rul["serial_number"].astype(str)

		original_training_data = original_training_data.sort_values(['serial_number', 'date']) # da togliere per usare il dummy dataset
		original_training_data.reset_index(drop=True, inplace=True)
		
		original_test_data = original_test_data.sort_values(['serial_number', 'date']) # da togliere per usare il dummy dataset
		original_test_data.reset_index(drop=True, inplace=True)
        
		original_training_data["serial_number"].replace(to_replace=r'_good', value='', regex=True, inplace=True)
		original_training_data = original_training_data.sort_values(["serial_number", "date"]).reset_index(drop=True)
        
		original_test_data["serial_number"].replace(to_replace=r'_good', value='', regex=True, inplace=True)
		original_test_data = original_test_data.sort_values(["serial_number", "date"]).reset_index(drop=True)

		sensors_to_ignore=['failure', 'serial_number', 'date'] 
		sensor_columns = [col for col in original_training_data.columns if (not col in sensors_to_ignore)]

		dfs_to_concatenate = [original_training_data]               
		for period in range(1, 1+num_aug_diff):
			g = partial(pd.DataFrame.diff, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_training_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'diff'+str(period)))
		for period in range(1, 1+num_aug_pctchange):
			g = partial(pd.DataFrame.pct_change, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_training_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'pctchange'+str(period)))
		original_training_data = pd.concat(dfs_to_concatenate, axis=1)
		temp_cols = list(original_training_data.columns)
		temp_cols.remove("failure")
		temp_cols.append("failure")
		original_training_data = original_training_data[temp_cols]

		dfs_to_concatenate = [original_test_data]               
		for period in range(1, 1+num_aug_diff):
			g = partial(pd.DataFrame.diff, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_test_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'diff'+str(period)))
		for period in range(1, 1+num_aug_pctchange):
			g = partial(pd.DataFrame.pct_change, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_test_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'pctchange'+str(period)))
		original_test_data = pd.concat(dfs_to_concatenate, axis=1)
		temp_cols = list(original_test_data.columns)
		temp_cols.remove("failure")
		temp_cols.append("failure")
		original_test_data = original_test_data[temp_cols]

		sensors_to_ignore=['failure', 'serial_number', 'date'] 
		sensor_columns = [col for col in original_training_data.columns if (not col in sensors_to_ignore)]
		
		# use correct type for numeric
		original_training_data[sensor_columns]=original_training_data[sensor_columns].astype(np.float64)
		original_test_data[sensor_columns]=original_test_data[sensor_columns].astype(np.float64)  

		
				
	elif  dataset_name in ("SMART_2016", "SMART_full"):
		def add_string_in_columns(df, string):
			#print("BAU")
			df.columns = [col.replace("smart", f"smart{string}") for col in df.columns]
			return df

		original_training_data = original_training_data.sort_values(['serial_number', 'date'])
		original_training_data.reset_index(drop=True, inplace=True)
		
		original_test_data = original_test_data.sort_values(['serial_number', 'date'])
		original_test_data.reset_index(drop=True, inplace=True)


		sensors_to_ignore=[]
		sensor_columns = [col for col in original_training_data.columns if ('smart' in col) and (not col in sensors_to_ignore)]

		dfs_to_concatenate = [original_training_data]               
		for period in range(1, 1+num_aug_diff):
			g = partial(pd.DataFrame.diff, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_training_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'diff'+str(period)))
		for period in range(1, 1+num_aug_pctchange):
			g = partial(pd.DataFrame.pct_change, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_training_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'pctchange'+str(period)))
		original_training_data = pd.concat(dfs_to_concatenate, axis=1)
		temp_cols = list(original_training_data.columns)
		temp_cols.remove("failure")
		temp_cols.append("failure")
		original_training_data = original_training_data[temp_cols]

		dfs_to_concatenate = [original_test_data]               
		for period in range(1, 1+num_aug_diff):
			g = partial(pd.DataFrame.diff, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_test_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'diff'+str(period)))
		for period in range(1, 1+num_aug_pctchange):
			g = partial(pd.DataFrame.pct_change, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_test_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'pctchange'+str(period)))
		original_test_data = pd.concat(dfs_to_concatenate, axis=1)
		temp_cols = list(original_test_data.columns)
		temp_cols.remove("failure")
		temp_cols.append("failure")
		original_test_data = original_test_data[temp_cols]

		sensors_to_ignore=[]
		sensor_columns = [col for col in original_training_data.columns if ('smart' in col) and (not col in sensors_to_ignore)]
		
		original_training_data[sensor_columns]=original_training_data[sensor_columns].astype(np.float64)
		original_test_data[sensor_columns]=original_test_data[sensor_columns].astype(np.float64)


	elif  dataset_name == "TEP":
		def add_string_in_columns(df, string):
			df.columns = [col.replace("xmeas", f"xmeas{string}").replace("xmv", f"xmv{string}") for col in df.columns]
			return df

		original_training_data.rename(columns={'simulationRun': 'serial_number',  'sample':'date'}, inplace=True)
		original_test_data.rename(columns={'simulationRun': 'serial_number',  'sample':'date'}, inplace=True)
		
		original_training_data = original_training_data.sort_values(['serial_number', 'date'])
		original_training_data.reset_index(drop=True, inplace=True)
		
		original_test_data = original_test_data.sort_values(['serial_number', 'date'])
		original_test_data.reset_index(drop=True, inplace=True)


		sensor_columns = [col for col in original_training_data.columns if ('xmeas' in col or 'xmv' in col) and not 'sax' in col]

		dfs_to_concatenate = [original_training_data]               
		for period in range(1, 1+num_aug_diff):
			g = partial(pd.DataFrame.diff, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_training_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'diff'+str(period)))
		for period in range(1, 1+num_aug_pctchange):
			g = partial(pd.DataFrame.pct_change, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_training_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'pctchange'+str(period)))
		original_training_data = pd.concat(dfs_to_concatenate, axis=1)
		temp_cols = list(original_training_data.columns)
		temp_cols.remove("failure")
		temp_cols.append("failure")
		original_training_data = original_training_data[temp_cols]

		dfs_to_concatenate = [original_test_data]               
		for period in range(1, 1+num_aug_diff):
			g = partial(pd.DataFrame.diff, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_test_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'diff'+str(period)))
		for period in range(1, 1+num_aug_pctchange):
			g = partial(pd.DataFrame.pct_change, periods=period)
			dfs_to_concatenate.append(add_string_in_columns(original_test_data.groupby("serial_number")[sensor_columns].apply(g).fillna(0.).replace([np.inf, -np.inf], 0), 'pctchange'+str(period)))
		original_test_data = pd.concat(dfs_to_concatenate, axis=1)
		temp_cols = list(original_test_data.columns)
		temp_cols.remove("failure")
		temp_cols.append("failure")
		original_test_data = original_test_data[temp_cols]

		sensor_columns = [col for col in original_training_data.columns if ('xmeas' in col or 'xmv' in col) and not 'sax' in col]
		
		original_training_data.rename(columns={sc:sc +"_" for sc in sensor_columns}, inplace=True)
		original_test_data.rename(columns={sc:sc +"_" for sc in sensor_columns}, inplace=True)
		
		sensor_columns = [col+'_' for col in sensor_columns ]  

		original_training_data[sensor_columns]=original_training_data[sensor_columns].astype(np.float64)
		original_test_data[sensor_columns]=original_test_data[sensor_columns].astype(np.float64)		
				
	
	else:
		assert False, "dataset non previsto"

    
	accepted_type = ['float']#['int', 'float']
	column_types={}
	for c in sensor_columns:
		column_types[c] = re.sub(r'[0-9]+', '', original_training_data[c].dtype.name)
		if not column_types[c] in  accepted_type:
			raise TypeError("Column '"+c+"' type not supported yet:" + column_types[c])
            
	#print(original_test_data.columns, original_training_data.columns, sensor_columns, column_types)


	## CONFIGURE HYPERPARAMETERS


	# margin_weight 
	# This parameter allows to customize the margin behaviour when separating positive and negative traces
	#  - If = 1, then the two margins are considered equal, thus the quest is for the best separating formula
	#  - If < 1, it gives predominance to the negative traces margin, and tries to extend it (lower FAR, lower recall)
	#     # 0.5 means it is considered 2x the importance of the positive one (margin distance gets divided by 2)
	#     # 0.25 means it is considered 4x, and so on (margin distance gets divided by 4)
	#  - If > 1, it gives predominance to the positive traces margin, and tries to extend it (higher far, higher recall)
	#     # 2 means it is considered 2x the importance of the negative one
	#     # 4 means it is considered 4x, and so on

	margin_weight = 1.0

	# Framework settings
	min_accuracy = params_dict["min_accuracy"] # used by the EA, a formula should have > of this value to be returned
	n_augs = params_dict["n_augs"] # genetic dataset will have n_augs*2 traces

	max_score_ea = params_dict["max_score_ea"]#0.1 # threshold for rule deactivation, removes a formula from the pool if exceeded
	max_score_pool = 0.99 # LEGACY - threshold for rule deactivation, removes a formula from the pool if exceeded
	score_beta = 0.9 # LEGACY - score forget rate (EWA)

	max_good_witnesses = -1
	good_witnesses_ratio = params_dict["good_witnesses_ratio"]
	max_bad_witnesses = 0.0
	bad_witnesses_ratio = 0.0
    
	remove_bad_good_from_training = False 
	num_augment_bad_good = 4


	# Random seeds generated with: np.random.randint(low=0, high=4294967295, size=10, dtype='uint32')
	seeds =  [996008960, 2264651732, 3144278649, 3742359108, 39449277,
			  521047483, 3679309865, 4181959796, 3975045990, 2806527179]
	seeds =  seeds[params_dict["num_seeds"]-1:params_dict["num_seeds"]]

    # Time series augmentation
	noise_scale = 0.01
	noise_kind = 'additive'


	# genetic programming algorithm settings
	max_gen=params_dict["max_gen"] # n. generations
	pop_size=params_dict["pop_size"] # population size
	cxpb=params_dict["cxpb"] # crossover prob for selected individualks
	mutpb=params_dict["mutpb"] # mutation probability for selected individualks
	max_horizon = 20
	ea_patience = params_dict["ea_patience"] # how many generations to wait for improvement before stopping
	mutation_decay = params_dict["mutation_decay"] # mutation decay rate
	mutate_all_or_one = 'all'

	exploit_idempotence = True
	cross_prob_terminal_node = 0.1 # 0.1 suggested by Koza (our defulat), 0.9 seems good for us (leads to 0 FAR?)

	return_multiple_formulas_ea = False

	processes=min(int((os.cpu_count() - 1)), pop_size) # how many individuals to evaluate at the same time
	print('Processes:', processes)


	#### only for debug
	#n_train_good_traces = 0 # for debug
	train_good_traces = []


	### FARs trainset ### 
	#n_train_far_traces = 200 # int(4*n_augs/5) # > 0 for debug only int(4*n_augs/5) # -1 => all2
	train_far_traces = []

		
	gp_instance = GP_STL(original_training_data, sensor_columns, processes, max_horizon, eval_wrapper, seed_in=42, exploit_idempotence=exploit_idempotence,
						cross_prob_terminal_node = cross_prob_terminal_node, max_height = params_dict['max_ea_height'], margin_weight = margin_weight, 
                        goodbad_traces_at_ea_generation=params_dict["goodbad_traces_at_ea_generation"], ea_score_method = 'FAR', 
                        ea_patience=ea_patience, mutation_decay=mutation_decay, mutate_all_or_one=mutate_all_or_one)


	debug=False
	dropout=.0 # only for debug




	training_set_repetitions = 1 # At least 1, only integer numbers are allowed
	early_stopping = False
	patience = 50  # Number of iterations to wait for improvement before stopping
	es_epsilon = 0.001 # Tolerance regarding the improvement 

	if is_test:
		train_res = load_pickle(train_res_path)
		test_res = np.full(len(seeds), None)
	else:
		train_res = np.full(len(seeds), None)

	## PREPARE AND EXECUTE TRAINING


	for s, seed in enumerate(seeds):        
		
		data = copy.deepcopy(original_training_data)
		test_data = copy.deepcopy(original_test_data)

		col_map_std = {}
        
		if dataset_name == "CMAPSS":

			random.seed(seed) 
			np.random.seed(seed) 
			
			CMAPSS_good_trace_prefix_ratio = 0.7
			
			CMAPSS_ratio_good_traces_training = -1.0
			
			trace_ids_training = data["serial_number"].unique()
			total_traces_training = len(trace_ids_training)
			if CMAPSS_ratio_good_traces_training == -1:
				good_traces_iter_training = trace_ids_training
				traces_to_concat = []
			else:
				good_traces_iter_training = np.random.choice(trace_ids_training, round(total_traces_training*CMAPSS_ratio_good_traces_training), replace=False).astype(str)
			data["failure"] = 0
			data.loc[data["serial_number"].isin(good_traces_iter_training), "failure"] = 1

			for trace_serial_id in trace_ids_training:
				if np.all(data[data["serial_number"] == trace_serial_id]["failure"].values == 0):
					len_trace = len(data[data["serial_number"] == trace_serial_id])
					cut_point = round(len_trace*CMAPSS_good_trace_prefix_ratio)
					data.drop(data[(data["serial_number"] == trace_serial_id ) & (data["date"] >= cut_point)].index, inplace=True)
				elif np.any(data[data["serial_number"] == trace_serial_id]["failure"].values == 0):
					assert False, "This should never happen"
				elif CMAPSS_ratio_good_traces_training == -1:
					new_good_trace = data[data["serial_number"] == trace_serial_id].copy()
					len_trace = len(new_good_trace)
					cut_point = round(len_trace*CMAPSS_good_trace_prefix_ratio)
					new_good_trace.drop(new_good_trace[new_good_trace["date"] >= cut_point].index, inplace=True)
					new_good_trace["failure"] = 0
					new_good_trace["serial_number"] += "_good"
					traces_to_concat.append(new_good_trace)

			if CMAPSS_ratio_good_traces_training == -1:
				temp = [data]
				temp.extend(traces_to_concat)
				data = pd.concat(temp, axis=0)

			data = data.reset_index(drop=True)

			
			trace_ids_test= test_data["serial_number"].unique()
			for trace_serial_id in trace_ids_test:
				len_trace = len(test_data[test_data["serial_number"] == trace_serial_id])
				total_len = test_rul[test_rul["serial_number"] == trace_serial_id]["RUL"].values[0] + len_trace
				if len_trace <= round(total_len * CMAPSS_good_trace_prefix_ratio):
					test_data.loc[test_data["serial_number"] == trace_serial_id, "failure"] = 0
				else:
					test_data.loc[test_data["serial_number"] == trace_serial_id, "failure"] = 1
		
			test_data = test_data.reset_index(drop=True)

		GP_STL.reverse_obj_window = data.groupby(by=["serial_number"], dropna=False).count().max().max()

		# Calculate the number of unique serial numbers to sample for validation
		num_to_sample = int(len(data['serial_number'].unique()) * 0.1)
		# Sample unique serial numbers
		sample_serial_numbers = data['serial_number'].drop_duplicates().sample(num_to_sample, replace=False)
		# Create a boolean mask for rows with the sampled serial numbers
		mask = data['serial_number'].isin(sample_serial_numbers)
		# Assign the selected rows to val_data
		val_data = data[mask].copy().reset_index(drop=True)
		
		# Remove the selected rows from the training data. This shall be done to esnure 90-10 split for HP tuning.
		# Uncomment if needed
		#data = data[~mask].reset_index(drop=True)

		if not is_test:
			train_res[s]=train_process((s, seed, data, max_good_witnesses, good_witnesses_ratio, training_set_repetitions, 
										early_stopping, patience, save_folder, dataset_name, output_pickle_file_suffix, es_epsilon, n_augs, sensor_columns, gp_instance, max_gen,
										   pop_size, cxpb, mutpb, max_score_ea, min_accuracy, column_types, score_beta, max_score_pool, val_data, return_multiple_formulas_ea,
                                           noise_scale, noise_kind, remove_bad_good_from_training, num_augment_bad_good, max_bad_witnesses, bad_witnesses_ratio, col_map_std))
		else:
			test_res[s]=test_process(s,seed,sensor_columns,column_types,params_dict["test_simulated_online"],test_data,data,train_res)


	## PRINT AND SAVE STATISTICS

	if not is_test:

		# eval stats
		precisions= [train_res[k]['precision'] for k in range(len(seeds))]
		recalls=[train_res[k]['recall'] for k in range(len(seeds))]
		fars=[train_res[k]['far'] for k in range(len(seeds))]
		f1_scores=[train_res[k]['f1_score'] for k in range(len(seeds))]
		
		print('Precisions', precisions)
		print('Avg', np.mean(precisions))
		
		print('Recalls', recalls) #recall
		print('Avg', np.mean(recalls))
		
		print('False allarm rates', fars)
		print('Avg', np.mean(fars))
		
		print('F1 scores',f1_scores)
		print('Avg', np.mean(f1_scores))


		precisions_full= [train_res[k]['precision_full'] for k in range(len(seeds))]
		recalls_full=[train_res[k]['recall_full'] for k in range(len(seeds))]
		fars_full=[train_res[k]['far_full'] for k in range(len(seeds))]
		f1_scores_full=[train_res[k]['f1_score_full'] for k in range(len(seeds))]
		
		print('Precisions_full', precisions_full)
		print('Avg_full', np.mean(precisions_full))
		
		print('Recalls_full', recalls_full) #recall
		print('Avg_full', np.mean(recalls_full))
		
		print('False allarm rates_full', fars_full)
		print('Avg_full', np.mean(fars_full))
		
		print('F1 scores_full',f1_scores_full)
		print('Avg_full', np.mean(f1_scores_full))


		# eval stats
		precisions_val= [train_res[k]['precision_val'] for k in range(len(seeds))]
		recalls_val=[train_res[k]['recall_val'] for k in range(len(seeds))]
		fars_val=[train_res[k]['far_val'] for k in range(len(seeds))]
		f1_scores_val=[train_res[k]['f1_score_val'] for k in range(len(seeds))]
		
		print('Precisions_val', precisions_val)
		print('Avg_val', np.mean(precisions_val))
		
		print('Recalls_val', recalls_val) #recall
		print('Avg_val', np.mean(recalls_val))
		
		print('False allarm rates_val', fars_val)
		print('Avg_val', np.mean(fars_val))
		
		print('F1 scores_val',f1_scores_val)
		print('Avg_val', np.mean(f1_scores_val))
		
		write_pickle(train_res, save_folder + 'train_stl_results_'+dataset_name + output_pickle_file_suffix +'.pickle')

	else:
		
		write_pickle(test_res, save_folder + 'test_stl_results_'+dataset_name + output_pickle_file_suffix +'.pickle')


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: script.py <path to train/test param dict> [test]")
        sys.exit(1)
    
    # Retrieve mandatory argument
    dict_path = sys.argv[1]
    
    # Retrieve optional argument or set a default value
    is_test = sys.argv[2] if len(sys.argv) > 2 else ""
    
    if is_test == "test":
    	is_test = True
    elif is_test == "":
    	is_test = False
    else:
    	assert False, "The second arguments must be either \"test\" or empty"
    	
    with open(dict_path, 'r') as file:
    	params_dict = eval(file.read())
    
    if is_test:
    	test_simulated_online = params_dict['test_simulated_online']
    	with open(params_dict['train_dict_path'], 'r') as file:
    		params_dict = eval(file.read())
    	params_dict['test_simulated_online'] = test_simulated_online
    
    
    main(params_dict, is_test)

