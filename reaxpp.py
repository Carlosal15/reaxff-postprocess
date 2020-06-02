#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:27:57 2020

@author: carlos
"""

import sys
import sys
sys.path.insert(0, '/Users/carlos/Work/TSM_CDT/PhD/reaxff-postprocess')
import os

import nw_initialiser
import networkx as nx 
def get_networks(bonds_filenames,G0,border_cutoff=0.3,mol_limit=200):
    """
    Parameters
    ----------
    bond_filenames : list of strings 
        list of bond dumps to process as strings _in a sorted list_. This is important
        because one may choose to write different files for different stages of a simulation
        but the postprocess takes some initial conditions into account, so they should be provided
        in the same order as they were produced.
    G0 : nx.Graph()
        the initial graph at step 0, with node attributes, obtained through Networkgen class
    border_cutoff : float
        bond order cutoff to use as criteria for considering to atoms bonded
    bdatafile : string
        path to the MAPS-generated lammps datafile containing the bonds. If None or false, the 
        starting bonds and molecules will be taken from the first timestep of the first 
        bond dump in bonds_filenames

    Returns
    -------
    networks: list of nx.Graph()
        list of graphs at each timestep in the bond dumps with the same node attributes
        as G0 (but no surf-surf bonding)
    tsteps: list of int
        timesteps

    """
    
    print('Cutoff:' +str(border_cutoff))
    
    NumAtoms=len(G0)
    
    
    networks=[]
    tsteps=[]
    for bonds_filename in bonds_filenames:
        print(bonds_filename)
        
        
        DataLines = open(bonds_filename, 'r').readlines()
        
        lnum=0
        while lnum<len(DataLines)-1:
            tstep=int(DataLines[lnum].split()[-1])
            tsteps.append(tstep)
            lnum=lnum+7
            
            G_ts = nx.Graph () #new graph with original attributes for this timestep
            G_ts.add_nodes_from (G0.nodes (data = True))
            
            AtomData={}
            AtomBonds={}
            bonds=[]
            
            timestep_lines=DataLines[lnum:lnum+NumAtoms]
            
            for i in range(NumAtoms):
                line = DataLines[lnum+i].split()
                
                try:
                    numbonds = int(line[2])
                except:
                    print(line)
                    print(lnum+i)
                linedata = tuple(map(float,line[:2*numbonds+4])) # saves atom IDs types num of bonds and bond orders                      
    
                AtomNum = linedata[0]
                
                for bond in range(numbonds):
                    #first 3 digits are index, type, nb, then index of bonded atoms, then molecule of the atom, then bond orders
                    if linedata[int(3+numbonds+1+bond)]>border_cutoff:
                                    bonds.append((AtomNum,linedata[int(3+bond)]))
                
            G_ts.add_edges_from(bonds)
            networks.append(G_ts)
            lnum=lnum+NumAtoms+1   
    return networks,tsteps

def write_bonds(list_networks,list_tsteps,attribute):
    """
    list_networks

    Parameters
    ----------
    list_networks : TYPE
        DESCRIPTION.
    list_tsteps : TYPE
        DESCRIPTION.
    attribute : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

                
if __name__ == "__main__":
    datafile='40xTSBP-2xFe3O4-scv.data'
    starting_bfile='40xTNBP-2xFe3O4-scv-BONDS.data'
    tal=nw_initialiser.Networkgen(datafile,starting_bfile)
    #G0=tal.get_equiv_labels(tal.G0)
    molgraphs =list(tal.G0.subgraph(c).copy() for c in nx.connected_components(tal.G0) if len(c)<200) 
    
    nx.draw_networkx(molgraphs[0],with_labels=True,labels=dict(molgraphs[0].nodes(data='label')))        
    bonds_filenames=['bonds_equil_500K-2GPa-10ms-40xTSBP-2xFe3O4.txt','bonds_heat_500K-2GPa-10ms-40xTSBP-2xFe3O4.txt','bonds_comp_500K-2GPa-10ms-40xTSBP-2xFe3O4.txt']  
    G=get_networks(bonds_filenames,tal.G0)      