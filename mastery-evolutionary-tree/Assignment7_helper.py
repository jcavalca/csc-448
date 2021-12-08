import numpy as np
import pandas as pd

import networkx as nx
import pandas as pd
import copy

import matplotlib.pyplot as plt

display_available = True
try:
    from IPython.display import Image
except:
    display_available = False
try:
    import pygraphviz
    graphviz_installed = True # Set this to False if you don't have graphviz
except:
    graphviz_installed = False

def draw(A):
    return Image(A.draw(format='png', prog='dot'))

patterns1 = ['ATAGA','ATC','GAT']
patterns2 = ['ananas','and','antenna','banana','bandana','nab','nana','pan']

# Inputs: G - networkx graph, current - node name, c - character on edge
# Output: a neighbor of current that is reached by an edge that has label==c; otherwise None
def find_edge(G,current,c): 
    for n in G.neighbors(current):
        if n is None:
            return None
        data = G.get_edge_data(current,n)
        if data['label'] == c:
            return n
    return None

def trie_construction(patterns):
    G = nx.DiGraph()
    G.add_node('root')
    node_count = 1
    for pattern in patterns:
        current = 'root'
        for n in pattern:
            node = find_edge(G, current, n)
            if not node:
                node = current
                G.add_edge(current, str(node_count), label=n)
                current = str(node_count)
                node_count += 1
            else:
                current = node
    return G

def trie_construction2(patterns):
    G = nx.DiGraph()
    G.add_node('root')
    node_count = 1
    for pattern in patterns:
        current = 'root'
        for n in pattern:
            node = find_edge(G, current, n)
            if not node:
                node = current
                G.add_edge(current, n, label=n)
                current = n
                node_count += 1
            else:
                current = node
    return G


def prefix_trie_matching(text,trie):
    symbol = ""
    v = "root"
    i = 0
    while i < len(text):
        if len(list(trie.neighbors(v))) == 0:
            return symbol
        else:
            w = find_edge(trie,v,text[i])
            if w is None:
                return None
            symbol += text[i]
            i += 1
            v = w
    return None

def trie_matching(text,trie):
    positions = []
    for i in range(len(text)):
        if prefix_trie_matching(text[i:], trie):
            positions.append(i)
    return positions

def suffix_trie(text):
    G = nx.DiGraph()
    G.add_node('root')
    node_count = 1
    current = 'root'
    for idx in range(len(text)):
        start = idx
        current = find_edge(G, 'root', text[idx])  
        
        if current:
            idx += 1
        else:
            current = 'root'
        
        for idx2 in range(idx, len(text)):
            if text[idx2] == '$':
                G.add_edge(current, str([start]), label=text[idx2])
            else:
                node = find_edge(G, current, text[idx2])
                if not node:
                    G.add_edge(current, str(node_count), label=text[idx2])
                    current = str(node_count)
                    node_count += 1
                else:
                    current = node
    return G

# Inputs: G - networkx graph, current - node name, c - character on edge
# Output: a neighbor of current that is reached by an edge that has label==c; otherwise None
def modified_find_edge(G,current,c):
    cv,j = c.split(",")
    j = int(j)
    for n in G.neighbors(current):
        if n is None:
            return None
        data = G.get_edge_data(current,n)
        cw,i = data['label'].split(",")
        i = int(i)
        if cw == cv and j > i:
            return n
    return None

def modified_suffix_trie(text):
    G = nx.DiGraph()
    G.add_node('root')
    leaf_nodes = []
    node_count = 1
    current = 'root'
    for idx in range(len(text)):
        start = idx
        c = f"{text[idx]}, {idx}"
        current = modified_find_edge(G, 'root', c)  

        if current:
            idx += 1
        else:
            current = 'root'
        
        for idx2 in range(idx, len(text)):
            label = f"{text[idx2]},{idx2}"
            if text[idx2] == '$':
                G.add_edge(current, str([start]), label=label)
                leaf_nodes.append(str([start]))
            else:
                node = modified_find_edge(G, current, label)
                if not node:
                    G.add_edge(current, str(node_count), label=label)
                    current = str(node_count)
                    node_count += 1
                else:
                    current = node
    return G,leaf_nodes


def collapse(trie, current, n1):
    c1, idx1 = trie.get_edge_data(current,n1)['label'].split(",")
    n2 = list(trie.neighbors(n1))[0]
    c2, idx2 = trie.get_edge_data(n1,n2)['label'].split(",")
    label = f"{c1+c2},{idx2}"
    trie.remove_edge(current, n1)
    trie.remove_edge(n1, n2)
    trie.remove_node(n1)
    trie.add_edge(current, n2, label=label)
    return n2

def suffix_tree_construction(text):
    trie,leaf_nodes = modified_suffix_trie(text)
    
    # creating a stack
    q = ['root']

    # while not empty
    while q: 
        current = q.pop()
        for n1 in sorted(trie.neighbors(current)):
            
            # keep collapsing when out_degree = 1
            while len(list(trie.neighbors(n1))) == 1:
                n1 = collapse(trie, current, n1)
                
            if len(list(trie.neighbors(n1))) > 1:
                for neighbor in trie.neighbors(n1):
                    deep_n1 = n1
                    
                    # keep collapsing when out_degree = 1 of neighbor node
                    while len(list(trie.neighbors(neighbor))) == 1:
                        deep_n1 = collapse(trie, deep_n1, neighbor)
                        neighbor = n1
                    q.append(neighbor)
    
    # changing edge labels
    for edge in list(trie.edges()):
        n1 = edge[0]
        n2 = edge[1]
        seq, idx = trie.get_edge_data(n1,n2)['label'].split(",")
        trie.remove_edge(n1, n2)
        trie.add_edge(n1, n2, label=seq)
        
    return trie

def to_adj(T):
    df = pd.DataFrame(nx.adjacency_matrix(T).todense(),index=T.nodes(),columns=T.nodes())
    for i in range(len(df)):
        for j in range(len(df)):
            if df.iloc[i,j] == 1:
                data = T.get_edge_data(df.index[i],df.columns[j])
                df.iloc[i,j] = data['label']
            else:
                df.iloc[i,j] = ""
    return df

def show(G):
    if graphviz_installed:
        # same layout using matplotlib with no labels
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
        #print(edge_labels)
        # Modify node fillcolor and edge color.
        #D.node_attr.update(color='blue', style='filled', fillcolor='yellow')
        #D.edge_attr.update(color='blue', arrowsize=1)
        A = nx.nx_agraph.to_agraph(G)
        # draw it in the notebook
        if display_available:
            display(draw(A))
        else:
            print(A)
    else:
        if display_available:
            display(to_adj(G))
        else:
            print(to_adj(G))
            
            