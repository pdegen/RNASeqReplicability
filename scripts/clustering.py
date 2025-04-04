from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def get_logCPM_linkage(df):

    distmethod = "ward"
    df_logCPM = find_logCPM(df, len(df), method="all", replacezeros=True)
    df_logCPM_corr = df_logCPM.corr()
    return hierarchy.linkage(df_logCPM_corr, method=distmethod)

def get_dendrogram(df,linkage,thresh=0,Plot=False, title='Hierarchical Clustering Dendrogram'):

    if Plot:
        plt.figure(figsize=(50, 20))

    dn = hierarchy.dendrogram(
        linkage,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        color_threshold=thresh,
        labels = df.columns,
        get_leaves = True,
        no_plot = not Plot
    )
    
    if Plot:
        plt.title(title)
        plt.xlabel('sample index')
        plt.ylabel('distance')
        plt.axhline(y=thresh, c='k', linestyle='--')
        #plt.savefig(f'{outpath}/{dataname}-dendrogram.png', facecolor='w', transparent=False)
        plt.show()

    return dn

def sort_cluster_colors(dn):
    """
    Sort leaves_color_list by index so that they can be clustered again by
    sns.clustermap and used as row colors
    """
    zipped = zip(dn["leaves_color_list"], dn["leaves"])
    zipped = sorted(zipped, key = lambda zipped: zipped[1])
    return [c[0] for c in zipped]

def display_clustered_logCPM_heatmap_NT(df, level=0, plot_dn=False, **kwargs):
    
    distmethod = "ward" #"average"
    vmin2, vmax2 = 0.7, 1
    vmin, vmax = 0.2, 1
    
    df_logCPM = find_logCPM(df, len(df), method="all", replacezeros=True)
    df_logCPM_corr = df_logCPM.corr()
    linkage = hierarchy.linkage(df_logCPM_corr, method=distmethod)
    
    thresh = 0.5#0.999*sorted(linkage[:,2], key=lambda x: -x)[level]
    print("Thresh", thresh)
    
    replicates = df.shape[1]//2
    fclusters = hierarchy.fcluster(linkage, t=thresh, criterion="distance")

    copy_T2N = True
    copy_N2T = False

    if copy_T2N:
        # Copy the T clusters to N clusters
        T_clusters = fclusters[replicates:]
        NT_clusters = np.concatenate([T_clusters,T_clusters])
        cluster_labels = np.unique(NT_clusters)
    elif copy_N2T:
        # Copy the T clusters to N clusters
        N_clusters = fclusters[:replicates]
        NT_clusters = np.concatenate([N_clusters,N_clusters])
        cluster_labels = np.unique(NT_clusters)
    else:
        NT_clusters = fclusters
        
    dn = get_dendrogram(df_logCPM_corr,linkage,thresh=thresh,Plot=plot_dn, title='Hierarchical Clustering Dendrogram')
    row_colors_cl = sort_cluster_colors(dn)
    if copy_T2N: row_colors_cl = row_colors_cl[replicates:] + row_colors_cl[replicates:] # Copy T clusters to N samples
    elif copy_N2T: row_colors_cl = row_colors_cl[:replicates] + row_colors_cl[:replicates] # Copy N clusters to T samples
    col_colors_cl = ["blue" for _ in range(1,replicates+1)] + ["red" for _ in range(1,replicates+1)]
    g = sns.clustermap(df_logCPM_corr,row_linkage=linkage,col_linkage=linkage,robust=True,row_colors=row_colors_cl,col_colors=col_colors_cl, vmin=vmin2, vmax=vmax2, method=distmethod)
    g.ax_row_dendrogram.axvline(thresh, c='magenta', linestyle='--',lw=1.5, alpha=0.5)
    return g, df_logCPM_corr


def find_logCPM(df, ntags, norm_factors = [], method="mean", replacezeros=False, value=1):

    if len(df) != ntags:
        raise Exception(f"Warning: Possibly logCPM calculated over subset of tags\nlen(df) = {len(df)}, tags = {ntags}")
  
    if replacezeros: df = df.replace(0,value)

    dfsum = df.sum()
    if norm_factors is not find_logCPM.__defaults__[0]:
        dfsum = (dfsum*norm_factors).astype(int)
  
    logCPM = np.log2(1e6*df.divide(dfsum))
    logCPM.replace([-np.inf,np.inf],np.nan,inplace=True)

    if method == "mean":
        return logCPM.mean(axis=1, skipna=True)
    return logCPM

def get_mean_corrs(df_logCPM_corr):
    df = df_logCPM_corr
    df_N = df.iloc[:len(df.columns)//2,:len(df.columns)//2]
    df_T = df.iloc[len(df.columns)//2:,len(df.columns)//2:]
    df_N = np.triu(df_N).flatten()
    df_T = np.triu(df_T).flatten()
    df_NT = np.triu(df).flatten() # actually NN, NT and TT corr
    return df_NT[df_NT>0].mean(), df_NT[df_NT>0].std(), df_N[df_N > 0].mean(), df_T[df_T > 0].mean(), df_N[df_N > 0].std(), df_T[df_T > 0].std()