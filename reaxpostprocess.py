#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:35:13 2020

@author: carlos
"""

import os
import os.path
import sys
import glob
#sys.path.append("/home/carlos/anaconda3/lib/python3.7/site-packages")
sys.path.append("/home/carlos/WORK/Phosphates_ReaxFF/python_scripts")
import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
import copy


#Generates list of networkx graphs for each tstep and adds nodes properties:
#species ('P','O'...) and labels ('P1','O2'...) to equivalent atoms in different
#molecules or equivalent sites of the same molecule. The later can be done
#by symmetry analyzing the 1st bond fix tstep molecules and symmetries, or
#with a datafile created with a classical FF that identifies the bonds 
#(sometimes reaxFF does not identify molecules as intended)

def get_species(datafile):
    
    """
    Parameters
    ----------
    datafile : string
        path to datafile (the one actually used in the simulation)

    Returns
    -------
    species : dictionary
        dictionary with atomic indexes (ints) as keys and atomic species (string)
        as items. E.g. {1:'C',2:'O',...}

    """
    
    lines=open(datafile,'r').read().splitlines()
    for i,line in enumerate(lines):
        if 'atoms' in line:
            NumAtoms=int(line.split()[0])
        if 'atom types' in line:
            NTypes=int(line.split()[0])
        if 'Atoms' in lines[i]:
            aindex=i+2
        if 'Masses' in lines[i]:
            mindex=i+2
    #generate an initial array of types just based on the Masses lines (which
    #doesn't have the labels used in the bonds)
    
    mlines=lines[mindex:mindex+NTypes]
    alines=lines[aindex:aindex+NumAtoms]
        
    species_masses={} #dict {1:'C',2:'O'...} according to the masses lines
    for line in mlines:
        species_masses[int(line.split()[0])]=line.split()[3]
    
    species={} #final dict that stores the indexes as keys and species (from masses) as items     
    for i,line in enumerate(alines):
          sp_ind=int(line.split()[1])
          ind=int(line.split()[0])
          species[ind]=species_masses[sp_ind]
    return species      
    

def mgrouper_bdatafile_nx(bonddatafile,mol_limit=200):
    """

    Parameters
    ----------
    bonddatafile : string
        path to the bond datafile
    mol_limit : int
        limit in the number of atoms to differentiate between surfaces and 
        reactant molecules

    Returns
    -------
    list_reactants : list of sets
        each set contains the indexes of atoms belonging to the same molecule
    list_surfs: list of sets
        same as above but for atoms in surfaces. NOTE: if there is not space
        between periodic replicas of upper and lower surfaces, the FF will think
        they're the same surface and this list will reflect so.
    bonds: networkx.classes.reportviews.EdgeView
        similar to a list of tuples with all the bonds (inc. surface);
        it's faster for iterating than an actual list

    """
    
    G = nx.Graph()
    
    lines=open(bonddatafile,'r').read().splitlines()
    
    for i,line in enumerate(lines):
        if 'atoms' in line:
            NumAtoms=int(line.split()[0])
        if 'bonds' in line:
            NumBonds=int(line.split()[0])
        if 'atom types' in line:
            NTypes=int(line.split()[0])
        if 'Bonds' in lines[i]:
            bindex=i+2
        if 'Atoms' in lines[i]:
            aindex=i+2
        if 'Masses' in lines[i]:
            mindex=i+2
     
        
    G.add_nodes_from(range(1,NumAtoms+1)) 
    blines=lines[bindex:bindex+NumBonds]
    alines=lines[aindex:aindex+NumAtoms]
    mlines=lines[mindex:mindex+NTypes]
    
    bonds_array=np.zeros([NumBonds,2])
    btuple_list=[]
    for  ind, line in enumerate(blines):
        #bond=np.array([int(line.split()[2]),int(line.split()[3])])
        #bonds_array[ind]=bond
        btuple_list.append([int(line.split()[2]),int(line.split()[3])])
    
    G.add_edges_from(btuple_list) #graph with all atoms and bonds
    bonds=copy.deepcopy(G.edges) #copy of all the bonds since they're removed from G in place later
    
    list_connected=list(nx.connected_components(G))
    
    list_surfs=[]
    
    for a in list_connected:
        if len(a)>mol_limit:
            list_surfs.append(a.copy())
    
    #remove surface(s) if present 
    if len(list_surfs)==1:
        G.remove_nodes_from(list(list_surfs[0]))
    elif len(list_surfs)>1:
        for surf in list_surfs:
            G.remove_nodes_from(list(surf))
    
    list_reactants=list(nx.connected_components(G))
    return list_reactants, list_surfs, bonds
def mgrouper_bdump_nx(bdump,border_cutoff=0.3,mol_limit=200):
    """

    Parameters
    ----------
    bdump : string
        path to the bond dump
    mol_limit : int
        limit in the number of atoms to differentiate between surfaces and 
        reactant molecules

    Returns
    -------
    list_reactants : list of sets
        each set contains the indexes of atoms belonging to the same molecule
    list_surfs: list of sets
        same as above but for atoms in surfaces. NOTE: if there is not space
        between periodic replicas of upper and lower surfaces, the FF will think
        they're the same surface and this list will reflect so.
    bonds: networkx.classes.reportviews.EdgeView
        similar to a list of tuples with all the bonds (inc. surface);
        it's faster for iterating than an actual list
    
    Get network data from the first timestep of the bonds dump file. Useful not
    to need a classical FF interpretation of the initial state.
    
    """
    G = nx.Graph()
    
    lines=open(bdump,'r').read().splitlines()
    
    NAtoms=int(lines[2].split()[-1])
    
    lines=lines[7:NAtoms+7]
    AtomData={}
    
    AtomBonds={}
    bonds=[]
    for i in range(NAtoms):
        line = lines[i].split()
        
        #print(lnum+i)
        numbonds = int(line[2])
                      
        linedata = tuple(map(float,line[:2*numbonds+4])) # saves atom IDs types num of bonds and bond orders                      

        AtomNum = linedata[0]
        
        #bonds=[]
        for bond in range(numbonds):
            #first 3 digits are index, type, nb, then index of bonded atoms, then molecule of the atom, then bond orders
            if linedata[int(3+numbonds+1+bond)]>border_cutoff:
                            bonds.append((AtomNum,linedata[int(3+bond)]))
                            
        
        AtomBonds[AtomNum]=bonds
        AtomData[AtomNum] = linedata[:]
        ###Discriminate the bonds in this step
    
    
    G = nx.Graph()
    G.add_nodes_from(AtomData.keys())
    G.add_edges_from(bonds)
    bonds=copy.deepcopy(G.edges) #copy of all the bonds since they're removed from G in place later
    
    list_connected=list(nx.connected_components(G))
    
    list_surfs=[]
    
    for a in list_connected:
        if len(a)>mol_limit:
            list_surfs.append(a.copy())
    
    #remove surface(s) if present 
    if len(list_surfs)==1:
        G.remove_nodes_from(list(list_surfs[0]))
    elif len(list_surfs)>1:
        for surf in list_surfs:
            G.remove_nodes_from(list(surf))
    
    list_reactants=list(nx.connected_components(G))
    return list_reactants, list_surfs, bonds
    


class Networkgen:
    
    def __init__(self, datafile,starting_bfile,border_cutoff=0.3,mol_limit=200):
        """

        Parameters
        ----------
        datafile : string
            Path to LAMMPS datafile used in the ReaxFF simulation
        starting_bfile : string
            File from which the initial molecules will be loaded, which is important
            to get all symmetries for equivalent atoms. Two options: either the text file
            from the reax/c/bonds fix, in which case the molecules will be taken from the
            1st timestep of the file; or a datafile generated with a classical FF (e.g.,
            with MAPS) that contains bonds to be employed (useful since reaxFF not 
            always identified the intended molecules)
            
        border_cutoff : float, optional
            Cutoff to consider two atoms bonded
        mol_limit : int, optional
            "Molecules" with more atoms than mol_limit will be considered surfaces and
            treated like one big atom

        Returns
        -------
        -

        """
        self.G_start=nx.Graph() #to store graph of initial/reference molecules
        self.species=get_species(datafile)
        with open(starting_bfile) as f:
            first_line = f.readline()
    
        if 'Timestep' in first_line: #bdump
            list_reactants, list_surfs, bonds=mgrouper_bdump_nx(starting_bfile,mol_limit=mol_limit,border_cutoff=border_cutoff) 
        else:
            list_reactants, list_surfs, bonds=mgrouper_bdatafile_nx(starting_bfile,mol_limit=mol_limit) 
            
        
            
            
            
            
            
            
            
            
            
            
            
            
            