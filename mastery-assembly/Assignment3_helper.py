import networkx as nx
import pandas as pd
import copy
from collections import Counter
import numpy as np

G = nx.MultiDiGraph()
G.add_edge(0,3)
G.add_edge(1,0)
G.add_edge(2,1)
G.add_edge(2,6)
G.add_edge(3,2)
G.add_edge(4,2)
G.add_edge(5,4)
G.add_edge(6,5)
G.add_edge(6,8)
G.add_edge(7,9)
G.add_edge(8,7)
G.add_edge(9,6)

G2 = nx.MultiDiGraph()
G2.add_edge(0,2);G2.add_edge(1,3);G2.add_edge(2,1);G2.add_edge(3,0);G2.add_edge(3,4);G2.add_edge(6,3);G2.add_edge(6,7);G2.add_edge(7,8);G2.add_edge(8,9);G2.add_edge(9,6)

kmers = ["CTTA","ACCA","TACC","GGCT","GCTT","TTAC"]

def calc_in_out(G):
    in_deg = {}
    out_deg = {}
    for u,v in G.edges():
        if v not in in_deg:
            in_deg[v] = 0
        if u not in out_deg:
            out_deg[u] = 0
        in_deg[v] += 1
        out_deg[u] += 1
    in_out = pd.Series(in_deg,name="in").to_frame().join(pd.Series(out_deg,name="out").to_frame(),how='outer')
    return in_out.fillna(0).astype(int)

def to_dense(graph):
    return nx.adjacency_matrix(graph).todense()

def to_adj(T):
    try:
        return pd.DataFrame(nx.adjacency_matrix(T).todense(),index=T.nodes(),columns=T.nodes())
    except:
        print("Cannot convert to adjacency matrix")
    return None

def show(T):
    T = copy.deepcopy(T)
    width_dict = Counter(T.edges())
    edge_width = [ (u, v, {'width': value}) 
                  for ((u, v), value) in width_dict.items()]
    
    G_new = nx.DiGraph()
    G_new.add_edges_from(edge_width)
    pos=nx.kamada_kawai_layout(G_new)
    #pos=nx.spring_layout(G_new)
    nx.draw(G_new, pos)
    edge_labels=dict([((u,v,),d['width'])
                 for u,v,d in G_new.edges(data=True)])
    
    nx.draw(G_new,pos,with_labels=True)
    nx.draw_networkx_edges(G_new, pos=pos)
    nx.draw_networkx_edge_labels(G_new, pos, edge_labels=edge_labels,
                                 label_pos=0.55, font_size=10)
    
def composition(k,text):
    patterns = []
    for i in range(len(text) - k + 1):
        patterns.append(text[i:i+k])
    patterns = sorted(patterns)
    return patterns

def de_bruijn(patterns):
    dB = nx.MultiDiGraph()
    # dB.add_edge("AA","AT") # sample edge in case you want to run the code without implementing your solution
    k = len(patterns[0])
    for kmer in patterns:
        prefix = kmer[:k-1]
        suffix = kmer[1:]
        dB.add_edge(prefix, suffix)
    return dB

def not_empty(arr):
    count = 0
    for i in arr:
        if (i[0] != None) and (i[1] != None):
            count += 1
    return count > 0

def eulerian_cycle(G,start=None):
    # YOUR SOLUTION HERE
    cycle = []
    
    # todo: we need a neighbors dictionary
    neighbors = {}
    edges = []
    for u, v in G.edges():
        if u not in neighbors:
            neighbors[u] = []
        neighbors[u].append(v)
        edges.append([u, v])
    # pick a node at random to start the cycle and add it to the cycle
    current = start if start else 0
    cycle.append(current)
    while not_empty(edges):
        current_neighbors = neighbors[current]
        # move around the cycle
        while not current_neighbors:
            for idx in range(len(cycle)):
                # pdb.set_trace()
                node = cycle[idx]
                if len(neighbors[node]) > 0:
                    new_cycle = cycle[idx::]
                    for idx2 in range(idx):
                        edges.append([cycle[idx2], cycle[idx2 + 1]])
                        neighbors[cycle[idx2]].append(cycle[idx2 + 1])
                        current = cycle[-1]
                    cycle = new_cycle 
                    current_neighbors = neighbors[current]
                    break
        v = current_neighbors.pop()
        cycle.append(v)
        edges.remove([current, v])
        current = v
    if cycle[-1] != current:
        cycle.append(current)
    return cycle

def eulerian_path(G):
    # YOUR SOLUTION HERE
    # call in_out on G
    # find the start and end nodes (keep track of those) 
    in_out_table = calc_in_out(G)
    start_node = None
    end_node = None
    show(G)
    for node, row in in_out_table.iterrows():
        in_degree = row['in']
        out_degree = row['out']
        if (in_degree + out_degree) % 2 == 1:
            if in_degree > out_degree: 
                start_node = node
            else:
                end_node = node
    # add a link between them such that G is now balanced
    G.add_edge(start_node, end_node)
    path = eulerian_cycle(G, start=end_node)
    return path

def reconstruct(kmers):
    dB = de_bruijn(kmers)
    path = eulerian_path(dB)
    text = path[0]
    for k in range(1, len(path) - 1):
        text += path[k][-1]
    # add each node prefix with the last character of the next sufix
    return text

def read_fasta(file):
    seqs = []
    headers = []
    # implement a read fasta file
    with open(file) as f:
        lines = f.readlines()
        chromosome = ""
        for line in lines:
            line = line.strip('\n')
            if line[0] == '>':
                if headers:
                    seqs.append(chromosome)
                    chromosome = ""
                headers.append(line)
            else: 
                chromosome += line
        seqs.append(chromosome)
    return headers,seqs

def align_dynamic2(s1,s2,verbose=False):
    scores = pd.DataFrame(index=["-"]+[s1[:i+1] for i in range(len(s1))],columns=["-"]+[s2[:i+1] for i in range(len(s2))])
    aligned = pd.DataFrame(index=["-"]+[s1[:i+1] for i in range(len(s1))],columns=["-"]+[s2[:i+1] for i in range(len(s2))])
    for s2_part in scores.columns:
        scores.loc["-",s2_part] = 0
        if s2_part == "-":
            aligned.loc["-","-"] = ("","")
        else:
            aligned.loc["-",s2_part] = ("".join(["-" for i in range(len(s2_part))]),s2_part)
    for s1_part in scores.index:
        scores.loc[s1_part,"-"] = 0
        if s1_part == "-":
            aligned.loc["-","-"] = ("","")
        else:
            aligned.loc[s1_part,"-"] = (s1_part,"".join(["-" for i in range(len(s1_part))]))
    if verbose:
        display(aligned)
    
    nrows,ncols = scores.shape
    for i in range(1,nrows):
        for j in range(1,ncols):
            # What are our three options
            opt1_s1 = aligned.index[i-1] # remember the rows are representative of s1
            opt1_s2 = aligned.columns[j-1] # remember the columns are representative of s2
            score = 0
            if scores.index[i][-1] == scores.columns[j][-1]:
                score_opt1 = scores.loc[opt1_s1, opt1_s2] + 1
            else: 
                score_opt1 = scores.loc[opt1_s1, opt1_s2]
            
            s1_aligned_opt1 = aligned.iloc[i-1, j-1][0] + scores.index[i][-1] 
            s2_aligned_opt1 = aligned.iloc[i-1, j-1][1] + scores.columns[j][-1]
            
            opt2_s1 = aligned.index[i-1]
            opt2_s2 = aligned.columns[j]
            score_opt2 = scores.loc[opt2_s1, opt2_s2]
            s1_aligned_opt2 = aligned.iloc[i-1, j][0] + scores.index[i][-1]
            s2_aligned_opt2 = aligned.iloc[i-1, j][1] + "-"

            opt3_s1 = aligned.index[i]
            opt3_s2 = aligned.columns[j-1]
            score_opt3 = scores.loc[opt3_s1, opt3_s2]
            s1_aligned_opt3 = aligned.iloc[i, j-1][0] + "-"
            s2_aligned_opt3 = aligned.iloc[i, j-1][1] + scores.columns[j][-1]
        
                
            # print(s1_aligned_opt1, s2_aligned_opt1, score_opt1)
            # print(s1_aligned_opt2, s2_aligned_opt2, score_opt2)
            # print(s1_aligned_opt3, s2_aligned_opt3, score_opt3)
            scores.loc[scores.index[i],scores.columns[j]] = max(score_opt1,score_opt2,score_opt3)
            if max(score_opt1,score_opt2,score_opt3) == score_opt1:
                aligned.loc[scores.index[i],scores.columns[j]] = (s1_aligned_opt1,s2_aligned_opt1)
            elif max(score_opt1,score_opt2,score_opt3) == score_opt2:
                aligned.loc[scores.index[i],scores.columns[j]] = (s1_aligned_opt2,s2_aligned_opt2)
            else:
                aligned.loc[scores.index[i],scores.columns[j]] = (s1_aligned_opt3,s2_aligned_opt3)
    if verbose:
        display(scores)
        display(aligned)
    return scores.loc[s1,s2],aligned.loc[s1,s2][0],aligned.loc[s1,s2][1]

def print_alignment(aligned_s1_,aligned_s2_,num_to_print=100):
    chunks_s1 = [aligned_s1_[i:i+num_to_print] for i in range(0, len(aligned_s1_), num_to_print)]
    chunks_s2 = [aligned_s2_[i:i+num_to_print] for i in range(0, len(aligned_s2_), num_to_print)]

    for aligned_s1,aligned_s2 in zip(chunks_s1,chunks_s2):
        for i in range(len(aligned_s1)):
            print(aligned_s1[i],end="")
        print()
        for i in range(len(aligned_s1)):
            if aligned_s1[i] == aligned_s2[i]:
                print("|",end="")
            else:
                print(" ",end="")
        print()
        for i in range(len(aligned_s2)):
            print(aligned_s2[i],end="")
        print()