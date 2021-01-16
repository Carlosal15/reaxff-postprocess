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
from pysmiles import write_smiles, fill_valence #to convert graphs to smiles
#from rdkit import Chem #to convert smiles to canonical as examplified below:
from collections import defaultdict
from pysmiles.write_smiles import  _get_ring_marker, _write_edge_symbol
from pysmiles.smiles_helper import remove_explicit_hydrogens, format_atom
from networkx import isomorphism
from indigo import *


#Generates list of networkx graphs for each tstep and adds nodes properties:
#species ('P','O'...) and labels ('P1','O2'...) to equivalent atoms in different
#molecules or equivalent sites of the same molecule. The later can be done
#by symmetry analyzing the 1st bond fix tstep molecules and symmetries, or
#with a datafile created with a classical FF that identifies the bonds 
#(sometimes reaxFF does not identify molecules as intended)


#TODO merge mgrouper_bdatafile_nx and mgrouper_bdump_nx into a single function


#Easy example to filter nodes by attribute: nodesP= [x for x,y in tal.G0.nodes(data='sp') if y=='P']
#Fast way to filter nodes that have an given attribute defined:  nx.get_node_attributes(G, 'attribute').keys()

#subgraphs =list(tal.G0.subgraph(c).copy() for c in nx.connected_components(tal.G0))




def write_smiles(molecule, default_element='*', start=None):
    """
    Creates a SMILES string describing `molecule` according to the OpenSMILES
    standard.
    Parameters
    ----------
    molecule : nx.Graph
        The molecule for which a SMILES string should be generated.
    default_element : str
        The element to write if the attribute is missing for a node.
    start : Hashable
        The atom at which the depth first traversal of the molecule should
        start. A sensible one is chosen: preferably a terminal heteroatom.
    Returns
    -------
    str
        The SMILES string describing `molecule`.
    """
    molecule = molecule.copy()
    #remove_explicit_hydrogens(molecule)

    if start is None:
        # Start at a terminal atom, and if possible, a heteroatom.
        def keyfunc(idx):
            """Key function for finding the node at which to start."""
            return (molecule.degree(idx),
                    # True > False
                    molecule.nodes[idx].get('element', default_element) == 'C',
                    idx)
        start = min(molecule.nodes, key=keyfunc)


    order_to_symbol = {0: '.', 1: '-', 1.5: ':', 2: '=', 3: '#', 4: '$'}

    dfs_successors = nx.dfs_successors(molecule, source=start)

    predecessors = defaultdict(list)
    for node_key, successors in dfs_successors.items():
        for successor in successors:
            predecessors[successor].append(node_key)
    predecessors = dict(predecessors)
    # We need to figure out which edges we won't cross when doing the dfs.
    # These are the edges we'll need to add to the smiles using ring markers.
    edges = set()
    for n_idx, n_jdxs in dfs_successors.items():
        for n_jdx in n_jdxs:
            edges.add(frozenset((n_idx, n_jdx)))
    total_edges = set(map(frozenset, molecule.edges))
    ring_edges = total_edges - edges

    atom_to_ring_idx = defaultdict(list)
    ring_idx_to_bond = {}
    ring_idx_to_marker = {}
    for ring_idx, (n_idx, n_jdx) in enumerate(ring_edges, 1):
        atom_to_ring_idx[n_idx].append(ring_idx)
        atom_to_ring_idx[n_jdx].append(ring_idx)
        ring_idx_to_bond[ring_idx] = (n_idx, n_jdx)

    branch_depth = 0
    branches = set()
    to_visit = [start]
    smiles = ''

    while to_visit:
        current = to_visit.pop()
        if current in branches:
            branch_depth += 1
            smiles += '('
            branches.remove(current)

        if current in predecessors:
            # It's not the first atom we're visiting, so we want to see if the
            # edge we last crossed to get here is interesting.
            previous = predecessors[current]
            assert len(previous) == 1
            previous = previous[0]
            if _write_edge_symbol(molecule, previous, current):
                order = molecule.edges[previous, current].get('order', 1)
                smiles += order_to_symbol[order]
        smiles += format_atom(molecule, current, default_element)
        if current in atom_to_ring_idx:
            # We're going to need to write a ring number
            ring_idxs = atom_to_ring_idx[current]
            for ring_idx in ring_idxs:
                ring_bond = ring_idx_to_bond[ring_idx]
                if ring_idx not in ring_idx_to_marker:
                    marker = _get_ring_marker(ring_idx_to_marker.values())
                    ring_idx_to_marker[ring_idx] = marker
                    new_marker = True
                else:
                    marker = ring_idx_to_marker.pop(ring_idx)
                    new_marker = False

                if _write_edge_symbol(molecule, *ring_bond) and new_marker:
                    order = molecule.edges[ring_bond].get('order', 1)
                    smiles += order_to_symbol[order]
                smiles += str(marker) if marker < 10 else '%{}'.format(marker)

        if current in dfs_successors:
            # Proceed to the next node in this branch
            next_nodes = dfs_successors[current]
            # ... and if needed, remember to return here later
            branches.update(next_nodes[1:])
            to_visit.extend(next_nodes)
        elif branch_depth:
            # We're finished with this branch.
            smiles += ')'
            branch_depth -= 1

    smiles += ')' * branch_depth
    return smiles





def graph_to_canonical_smiles(G,allHsExplicit=True):
    #yields a canonical string from a graph representing a molecule, with
    #node attributes 'element'
    sm=write_smiles(G)
    indigo = Indigo()
    
    mol=indigo.loadMolecule(sm)
    mol.aromatize()
    return mol.canonicalSmiles() #Does not include with H...
    
    
    #return Chem.MolToSmiles(m,isomericSmiles=False,allHsExplicit=allHsExplicit)

def elem_match(dict1,dict2):
    #match species for isomorphism tests
    return dict1['element']==dict2['element']

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
    full_G: networx.Graph
        The full starting network (only with atomic indexes)

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
    
    full_G=copy.deepcopy(G)
    
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
    return list_reactants, list_surfs, bonds,full_G

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
    full_G: networx.Graph
        The full starting network (only with atomic indexes)
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
    
    full_G=copy.deepcopy(G)
    
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
    return list_reactants, list_surfs, bonds, full_G
    


class Networkgen:
    
    def __init__(self, datafile,starting_bfile,border_cutoff=0.3,mol_limit=100):
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
        
        
        #Get species
        self.species=get_species(datafile)
        self.mol_limit=mol_limit
        self.datafile=datafile
        self.starting_bfile=starting_bfile
        
        #Get starting graph and separate reactant molecules and surfaces
        with open(starting_bfile) as f:
            first_line = f.readline()
    
        if 'Timestep' in first_line: #bdump
            self.list_reactants, self.list_surfs, self.bonds, self.G0=mgrouper_bdump_nx(starting_bfile,mol_limit=mol_limit,border_cutoff=border_cutoff) 
        else:
            self.list_reactants, self.list_surfs, self.bonds, self.G0=mgrouper_bdatafile_nx(starting_bfile,mol_limit=mol_limit) 
         
        self.surfatoms=set()    
        for surf in self.list_surfs:
            for at in surf:
                self.surfatoms.add(at)
                
        self.reactants=set()    
        for m in self.list_reactants:
            for at in m:
                self.reactants.add(at)
            
        #Add element-related attributes to nodes and edges
        for node in self.G0.nodes:
            self.G0.add_nodes_from([node], element=self.species[node]) #changed label of attribute 'sp' to 'element'
        
        #Add sorted element attributes to bonds (eg O-P)
        for bond in self.G0.edges:
            sp1=self.species[bond[0]]
            sp2=self.species[bond[1]]
            spinbond=[sp1,sp2]
            spinbond.sort()
            self.G0.add_edges_from([bond],sp= '-'.join(spinbond))
        
        self.G0=self.get_equiv_labels(self.G0,mol_limit)    
            
    def get_equiv_labels(self, G0,mol_limit=200):
        """

        Parameters
        ----------
        G0 : nx.Graph
            nx.Graph containing atomic indexes as nodes and bonds as edges, 
            with species as 'sp' attribute for each node

        Returns
        -------
        G0 : nx.Graph
            adds equivalent label attributes to the nodes (e_g: P1, O2...)
        list_equivs: list of lists
            each nested list contains indexes of equivalent atoms, so that
            it's easy to relabel/reassign attributes manually to the network
        
        Finds which atoms are in equivalent positions up to 3rd neighbors to
        assign them the same labels and easily identify which bonds break in a
        reaxFF simulation.
          
        If a connected component of a graph has more molecules than mol_limit, 
        it is considered a surface and atoms in it are only assigned an attribute
        
        """
        
        #generate all connected graphs of molecules
        molgraphs =list(G0.subgraph(c).copy() for c in nx.connected_components(G0) if len(c)<mol_limit)
        
        #graphs of surfaces    
        surfgraphs =list(G0.subgraph(c).copy() for c in nx.connected_components(G0) if len(c)>=mol_limit)
            
        #Find which molecules are the same (check isomorphism and species)
        
        checked_mols=[]
        equiv_mols=[]
        
        sm_list=[]
        
        for i in range(len(molgraphs)):
            if i not in checked_mols:
                newmol_type=[i]
                
                for j in range(i+1,len(molgraphs)):
                    GM = isomorphism.GraphMatcher(molgraphs[i],molgraphs[j],node_match=self.elem_match)
                    if GM.is_isomorphic():
                        newmol_type.append(j)
                checked_mols.extend(newmol_type)
                equiv_mols.append(newmol_type)    
                sm_list.append(graph_to_canonical_smiles(molgraphs[i],allHsExplicit=True))
        #
        
        print('There are the following compounds and quantities:')
        for i,m in enumerate(equiv_mols):
            print(str(len(m))+' : '+sm_list[i]) #Does not include H but it's an easy way to check different molecules
        
        start=time.time()
        dict_neighborpaths={}
        for mol_type in equiv_mols:
            #Assign a string to each atom based on sorted paths to 2nd neighbours
            #store paths in dict to then compare and assign eq labels to all atoms with same paths
            for mol in mol_type:
                mymol=molgraphs[mol]
                
                for node in list(mymol):
                    paths=[]
                    sp0=mymol.nodes[node]['element'] 
                    for neigh in mymol.neighbors(node):
                        sp1=mymol.nodes[neigh]['element'] 
                        
                        noneigh2=True #condition in case there are no 2nd neighbours
                        for neigh2 in mymol.neighbors(neigh):
                            
                            if neigh2 != node:
                                noneigh2=False
                                sp2=mymol.nodes[neigh2]['element'] 
                                paths.append(sp0+sp1+sp2)
                            
                        if noneigh2: #append only 1st neighbours if there aren't 2nd ones
                            paths.append(sp0+sp1) 
                            
                    paths.sort()
                    dict_neighborpaths[node]=paths
        print(time.time()-start)
        
        
        #Since I want the script to reproduce the same labels for the same atoms (paths),
        #need to iterate through alphabetically sorted paths instead of the randomly sorted
        #indexes. For that purpose we use the dict:
        
        dict_paths={} #'pathstring':[ind1,ind2]
        
        for key,value in  dict_neighborpaths.items():
            pathstring='-'.join(value)
            if pathstring not in dict_paths:
                dict_paths[pathstring]=[key]
            else:
                dict_paths[pathstring].append(key)
        
        dict_pathssort={}
        for key, value in sorted(dict_paths.items()):
            dict_pathssort[key]=value
        
        #Now iterate over this dictionary and assign labels in that order
        labs_used=[]
        for path,indices in dict_pathssort.items():
            elem=G0.nodes[indices[0]]['element']
            ind=1
            while elem+'_'+str(ind) in labs_used:
                ind=ind+1
            newlab=elem+'_'+str(ind)
            labs_used.append(newlab)
            for atom in indices:
                G0.nodes[atom]['label']=newlab
        
        for ind, surf in enumerate(surfgraphs):
            if len(surfgraphs)<=1:
                for atom in surf:
                    elem=G0.nodes[atom]['element']
                    G0.nodes[atom]['label']=elem+'_surf'
            else:
                for atom in surf:
                    elem=G0.nodes[atom]['element']
                    G0.nodes[atom]['label']=elem+'_surf'+str(ind+1)
                
        
        return G0
    
    def get_networks(self,bonds_filenames,border_cutoff=0.3,mol_limit=200):
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
        
        NumAtoms=len(self.G0)
        
        
        networks=[]
        tsteps=[]
        for bonds_filename in bonds_filenames:
            print(bonds_filename)
            
            
            DataLines = open(bonds_filename, 'r').readlines()
            
            lnum=0
            while lnum<len(DataLines)-1:
                tstep=int(DataLines[lnum].split()[-1])
                #print(str(lnum)+' '+DataLines[lnum])
                tsteps.append(tstep)
                lnum=lnum+7
                
                G_ts = nx.Graph () #new graph with original attributes for this timestep
                G_ts.add_nodes_from (self.G0.nodes (data = True))
                
                #AtomData={}
                #AtomBonds={}
                bonds=[]
                
                #timestep_lines=DataLines[lnum:lnum+NumAtoms]
                
                #for i in range(NumAtoms):
                while DataLines[lnum][0] !='#':
                    line = DataLines[lnum].split()
                    
                    try:
                        numbonds = int(line[2])
                    except:
                        print(line)
                        print(lnum)
                    try:
                        linedata = tuple(map(float,line[:2*numbonds+4])) # saves atom IDs types num of bonds and bond orders        
                    except:
                        print('Error in: '+bonds_filename)
                        print('line: '+str(lnum))
                        print(line)
        
                    AtomNum = linedata[0]
                    #TODO parse only reactants, will be quicker
                    for bond in range(numbonds):
                        #first 3 digits are index, type, nb, then index of bonded atoms, then molecule of the atom, then bond orders
                        
                        
                        #comment out the following conditionto track surface bonds as well
                        #if any(at in self.reactants for at in [AtomNum,linedata[int(3+bond)]]):
                            if linedata[int(3+numbonds+1+bond)]>border_cutoff:
                                            bonds.append((AtomNum,linedata[int(3+bond)]))
                    
                    lnum=lnum+1
                    
                G_ts.add_edges_from(bonds)
                networks.append(G_ts)
                lnum=lnum+1
                #lnum=lnum+NumAtoms+1   
        self.networks=networks
        self.tsteps=tsteps
        #return networks,tsteps
        
    def write_bonds(self,attribute='label',fout='bonds_pp.txt'):
        """
        list_networks
    
        Parameters
        ----------
        G0: nx.Graph()
            reference starting bonds
        networks : list of graphs
            list of graphs corresponding to a dump file (generated by get_networks)
        tsteps : list of ints
            corresponding timesteps for the graphs
        attribute : str
            node attributes to be used for bond counting. Eg, if 'element', it 
            will count the number of P-O bonds and so on; if 'label', P_1-O_2... 
            other attributes can be added to the nodes to be employed with this function
    
        Generates a file fout with a header and lines with bonds of each tipe per timestep, eg:
        Timestep	Fe-C1	Fe-C2	Fe-C3	Fe-C4	Fe-O1	Fe-O2	O1-P1 O2-P1	O1-C1	O1-C2	O1-C3	O1-C4	O2-C1	O2-C2	O2-C3	O2-C4
       	4000	0	0	0	0	0	0	0	0	0	0	0	0	0	0
    
        """
        
        #Get all unique values of the corresponding attribute
        values=set()
        for key, value in nx.get_node_attributes(self.G0,attribute).items():
            values.add(value)
        values=list(values)
        #bonds will be stored in a dictionary, sort the keys:
        
        header="Timestep"+"\t"
        dict_bonds={} #to store bonds, as 'att1-att2:nbonds'
        for i,val1 in enumerate(values):
            for j,val2 in enumerate(values[i:]):
                key=[val1,val2]
                #key.sort()
                key='-'.join(sorted(key))
                dict_bonds[key]=np.zeros(len(self.tsteps))
                header=header+key+"\t"
        header=header+"\n"
        
        bfile=open(fout,'w')
        bfile.write(header)
        
        for i,tstep in enumerate(self.tsteps):
            network=self.networks[i]
            bline=str(tstep)+"\t"
            for edge in network.edges:
                atr0=network.nodes[edge[0]][attribute]
                atr1=network.nodes[edge[1]][attribute]
                key=[atr0,atr1]
                #key.sort()
                key='-'.join(sorted(key))
                dict_bonds[key][i]=dict_bonds[key][i]+1
        
            for key in dict_bonds:
                bline=bline+str(int(dict_bonds[key][i]))+"\t"
            bline=bline+"\n"
            bfile.write(bline)
    
    def ovito_mols(self,fout='Ovito_molind.txt'):
        molgraphs =list(self.G0.subgraph(c).copy() for c in nx.connected_components(self.G0) if len(c)<self.mol_limit)
        f=open(fout,'w')
        counter=1
        for graph in molgraphs:
            f.write('Molecule: '+str(counter)+'\n')
            counter=counter+1
            for node in list(graph)[:-1]:
                f.write('ParticleIdentifier=='+str(node)+'||\n')
            f.write('ParticleIdentifier=='+str(list(graph)[-1])+'\n\n')
        f.close()
            
    ########################################################################
    # WRITE BOND DATAFILES FUNCTION HERE
    ########################################################################


    def write_bonds_datafiles(self, dumpfiles,path_name_data_out,datafile_for_header):
        """
    
        Parameters
        ----------
        dumpfiles: list
            list with the names of the dumpfiles, in order
        path_name_data_out: string
            path to where the bond datafiles will be created, with the name of the datafiles, to which the timesteps will be added. E.g.
            '/folder/datafout' may create files in the folder 'folder' that will be named datafout1000, datafout2000... this syntax is needed
            for Ovito
        datafile_for_header: string
            path to datafile from which the header (lines until Masses are stated, inclusive) is taken for the creating the other datafiles
        Returns
        ----------
        Creates bond datafiles, one per timestep.

        In order to create a live visualization of the reaxff simulation bonds each timestep, this function creates lammps datafiles with the 
        corresponding bonding information for each timestep. These datafiles can be read like a dumpfile in ovito, but now including the reaxff bonds.
        Ovito identifies file names patterns that may differ by a number, and assumes that number is the timestep, so that it sorts them and reads them as a single file. 
    
        """
        #From the following is read the first few lines which contains numbers of atoms, bonds, etc... supercell and mass info, until the line Atoms [so Atoms should be the 1st info after this 'header']
        #datafile_for_header="/Users/carlos/Documents/tests/ovito_datafile_bonds/TSBP_Fe_x48.data" 
        
        
        header_lines=open(datafile_for_header,'r').readlines()

        #Find the line index of the first empty line after specifying the masses. That's the limit of the header
        #The index is the second empty string after Masses (the first one is the empty one immediately after it)
        pass_masses=False
        pass_first_empty=False
        for index,line in enumerate(header_lines):
            if 'Masses' in line:
                pass_masses=True
            elif pass_masses and not line.split():
                pass_first_empty=True
            elif pass_masses and not line.split() and pass_first_empty:
                index=index
                break
        header_lines=header_lines[:index]  #the index was 26 by default in most cases, but the way here is more versatile


        for file in dumpfiles:
            #dumpfile='dump_comp.lammpstrj'
            dumplines=open(file,'r').readlines()

            #name_data_out='/Users/carlos/Documents/tests/ovito_datafile_bonds/data_bonds_out/400K-4GPa-48xTSBP-2xFe_'
            #needs the networks generated from the reax_pp
            dump_index=0
            dump_tsteps=[]
            while dump_index+1 <= len(dumplines): 
            #for index, network in enumerate(self.networks):
                
                dump_tstep=int(dumplines[dump_index+1].split()[0])
                dump_tsteps.append(dump_tstep)
                
                dump_natoms=int(dumplines[dump_index+3].split()[0])
                
                
                
                #read positions: 
                dict_pos={} #keys are indexes, items are positions 
                for line in dumplines[dump_index+9:dump_index+9+dump_natoms]:
                    dict_pos[line.split()[0]]=' '.join(line.split()[1:])
                
                
                
                dump_index=dump_index+9+dump_natoms
                
                data_out=path_name_data_out+str(dump_tstep)+'.data'
                
                nbonds=len(self.networks[self.tsteps.index(dump_tstep)].edges)
                #change the 'bonds' line in the header
                for line_ind,line in enumerate(header_lines):
                    if 'bonds' in line:
                        header_lines[line_ind]=str(nbonds)+' bonds \n'
                    if 'atoms' in line:
                        header_lines[line_ind]=str(dump_natoms)+' atoms \n'
                
                my_new_lines=header_lines.copy()
                
                
                #Add position info
                my_new_lines.append('Atoms\n\n')
                for atom in dict_pos:
                    #lines in bond style datafile: atom-ID molecule-ID atom-type x y z
                    #whereas lines in dump are id type x y z 
                    at_str=atom+' 1 '+dict_pos[atom]+'\n'
                    my_new_lines.append(at_str)
                
                #Add bonds info
                my_new_lines.append('\nBonds\n\n')
                bcounter=1
                for bond in tal.networks[tal.tsteps.index(dump_tstep)].edges:
                    b_str=str(bcounter)+' 1 '+str(int(bond[0]))+' '+str(int(bond[1]))+'\n'
                    bcounter=bcounter+1
                    my_new_lines.append(b_str)
                
                dout=open(path_name_data_out+str(dump_tstep),'w')
                dout.writelines(my_new_lines)
                dout.close()

            
    def elem_match(self,dict1,dict2):
        #match species for isomorphism tests
        return dict1['element']==dict2['element']     
if __name__ == "__main__":
    start=time.time()
    attribute='element'
    datafile='calcite-diamond.data'
    starting_bfile='bonds_equil.txt'
    tal=Networkgen(datafile,starting_bfile)
    #G0=tal.G0
    tinit=time.time()
    print('Initialised in:')
    print(tinit-start)
    #nx.get_node_attributes(G0,attribute)
    molgraphs =list(tal.G0.subgraph(c).copy() for c in nx.connected_components(tal.G0) if len(c)<200) 
    tdraw=time.time()
    print('Drawn in:')
    print(tdraw-tinit)
    #nx.draw_networkx(molgraphs[0],with_labels=True,labels=dict(molgraphs[0].nodes(data='label')))        
    bonds_filenames=['bonds_equil.txt','bonds_comp.txt']  
    #networks,tsteps=get_networks(bonds_filenames,tal.G0)    
    tal.get_networks(bonds_filenames)
    tprocess=time.time()
    print('Generated networks in')
    print(tprocess-tdraw)
    tal.write_bonds(fout='btest_pp.txt')
    twrite=time.time()
    print('Wrote bonds in:')
    print(twrite-tprocess)  

    directory_ovito_out='[input directory here]'
    path_name_data_out='[path to datafiles to be created; I suggest end it with an underscore, after which each tstep number will be appended]'
    #create directory if it doesn't exist
    try:
        os.mkdir('/home/carlos/WORK/Phosphates_ReaxFF/ReaxFF/Comp_shear/bonds_datafile_out_OVITO/'+system)
    except:
        pass

    datafile_for_header=datafile
    tal.write_bonds_datafiles(dumps_filenames,path_name_data_out,datafile_for_header)
            