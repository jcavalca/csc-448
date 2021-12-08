import pandas as pd
import numpy as np

def print_alignment(aligned_s1,aligned_s2):
    num_to_print=len(aligned_s1)
    for i in range(num_to_print):
        print(aligned_s1[i],end="")
    print()
    for i in range(num_to_print):
        if aligned_s1[i] == aligned_s2[i]:
            print("|",end="")
        else:
            print(" ",end="")
    print()
    for i in range(num_to_print):
        print(aligned_s2[i],end="")

def align_dynamic3(s1,s2,match_score=1,mismatch_score=0,gap_score=0,verbose=False):
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
            opt1_s1 = scores.index[i-1] # remember the rows are representative of s1
            opt1_s2 = scores.columns[j-1] # remember the columns are representative of s2
            score = 0
            if scores.index[i][-1] == scores.columns[j][-1]:
                score_opt1 = scores.loc[opt1_s1, opt1_s2] + match_score
            else: 
                score_opt1 = scores.loc[opt1_s1, opt1_s2] + mismatch_score
            
            s1_aligned_opt1 = aligned.iloc[i-1, j-1][0] + scores.index[i][-1] 
            s2_aligned_opt1 = aligned.iloc[i-1, j-1][1] + scores.columns[j][-1]
            
            opt2_s1 = scores.index[i-1]
            opt2_s2 = scores.columns[j]
            score_opt2 = scores.loc[opt2_s1, opt2_s2] + gap_score
            s1_aligned_opt2 = aligned.iloc[i-1, j][0] + scores.index[i][-1]
            s2_aligned_opt2 = aligned.iloc[i-1, j][1] + "-"
            
            opt3_s1 = scores.index[i]
            opt3_s2 = scores.columns[j-1]
            score_opt3 = scores.loc[opt3_s1, opt3_s2] + gap_score
            s1_aligned_opt3 = aligned.iloc[i, j-1][0] + "-"
            s2_aligned_opt3 = aligned.iloc[i, j-1][1] + scores.columns[j][-1]
            
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
