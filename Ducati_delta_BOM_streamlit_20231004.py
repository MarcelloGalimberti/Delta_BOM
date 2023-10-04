#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from PIL import Image # serve per l'immagine?
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[279]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))


# ---

# In[2]:


def load_data(url, index=''):
    if index:
        data = pd.read_excel(url, index_col=index)
    else:
        data = pd.read_excel(url)
    return data


# In[3]:


url_SAP = 'https://github.com/MarcelloGalimberti/Delta_BOM/blob/main/SAP%20CS12-D33008424-DB10-06.11.2023-Capg.XLSX?raw=true'
url_PLM = 'https://github.com/MarcelloGalimberti/Delta_BOM/blob/main/MBOM_DB10_STIP_EUR_RED_A%20RAGGI_20231003-Filtrato_isSpare.xlsx?raw=true'


# ## Importo ed elaboro file Cap

# In[4]:


SAP_cap = load_data(url_SAP)


# In[6]:


SAP_cap['Liv.']=SAP_cap['Liv. esplosione'].str[-1]


# In[7]:


SAP = SAP_cap.drop(columns=['Liv. esplosione','Numero posizione','Testo breve oggetto',
                           'Unità misura base','Tipo pos.','Testo posizione riga 1',
                           'Testo posizione riga 2','Tipo di materiale','Intra material',
                           'Indice di rilevanza del CCST'])


# In[8]:


SAP = SAP[['Liv.','Numero componenti','Qtà comp. (UMB)','Merce sfusa']]


# In[9]:


SAP.rename(columns={'Numero componenti':'Articolo','Qtà comp. (UMB)':'Qty'},inplace=True)


# In[10]:


SAP = SAP.fillna(0)


# In[11]:


SAP['Merce sfusa'] = SAP['Merce sfusa'].replace('X',1)


# In[12]:


SAP['Liv.']= SAP['Liv.'].astype(int)


# ### Algo per eliminare merce sfusa in file Cap

# In[14]:


SAP_working = SAP.copy()


# In[15]:


SAP_working['Eliminare'] = 0


# In[16]:


for i in range(len(SAP_working)):
    if i == len(SAP_working):
        break
    if SAP_working.loc[i,'Merce sfusa'] == 1:
        livello_padre = SAP_working.loc[i,'Liv.']
        SAP_working.loc[i,'Eliminare'] = 1
        j = i
        if (j+1) == len(SAP_working):
            break
        while SAP_working.loc[j+1,'Liv.']>livello_padre:
            SAP_working.iat[j+1,4]=1
            j+=1
            if (j+1) == len(SAP_working):
                break


# In[17]:


SAP_no_merce_sfusa = SAP_working.loc[SAP_working['Eliminare']==0]


# In[18]:


SAP_no_merce_sfusa = SAP_no_merce_sfusa.drop(columns=['Merce sfusa','Eliminare'])


# In[19]:


SAP_no_merce_sfusa.reset_index(drop=True, inplace=True)


# ---

# ## Importo ed elaboro file Siemens

# In[22]:


PLM = load_data(url_PLM)


# In[23]:


PLM.columns


# In[25]:


PLM_reduced = PLM.drop(columns=['Home', 'Level.1', 'BOM Line', 'Release Statuses',
       'Occurrence Effectivities', 'Effectivity Formula', 'Rev Name',
       'Variant Formula','ID In Context (All Levels)', 'Find No.',
       'Revision Effectivity', 'Unit Of Measure', 'Position Type', 'View Type',
       'Item Type', 'Revision', 'Application Description', 'Text Note 2',
       'Descriptive Connection', 'Is Spare Part', 'Disassemble','Date Released'])


# In[26]:


PLM_reduced.rename(columns={'Level':'Liv.','Item Id':'Articolo','Quantity':'Qty'},inplace=True)


# In[27]:


PLM_reduced.fillna(1,inplace=True)


# ---

# ## Ottengo working file

# In[28]:


# Rinomino entrambi i file
PLM_BOM = PLM_reduced.copy()
SAP_BOM=SAP_no_merce_sfusa.copy()


# In[29]:


# salva in xlsx
PLM_BOM.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Delta BOM/Confronto Python/PLM_BOM.xlsx',
                index=False)
SAP_BOM.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Delta BOM/Confronto Python/SAP_BOM.xlsx',
                index=False)


# ## Statistiche BOMs e pubblicazione Streamlit

# ### Lista livelli 1  - serve per sottoalberi M, V, X

# In[30]:


SAP_livelli_1 = SAP_BOM[SAP_BOM['Liv.'] == 1]
lista_SAP_livelli_1 = SAP_livelli_1.Articolo.to_list()


# In[31]:


PLM_livelli_1 = PLM_BOM[PLM_BOM['Liv.'] == 1]
lista_PLM_livelli_1 = PLM_livelli_1.Articolo.to_list()


# In[32]:


lista_SAP_livelli_1


# In[33]:


lista_PLM_livelli_1


# ### Script per estrarre il sottoalbero delle BOM a partire da un SKU

# In[34]:


# Inserire SKU e BOM nella funzione partizione


# In[35]:


def partizione (SKU,BOM):
    indice = BOM.index[BOM['Articolo'] == SKU].tolist()
    idx=indice[0]
    livello_SKU = BOM.iloc[idx,0]
    j=idx+1
    if j == len(BOM):
        indice_target = idx
        df = BOM.iloc[idx:indice_target+1,:]
    elif BOM.iloc[j,0] <= livello_SKU:  
        indice_target = j
        df = BOM.iloc[idx:indice_target,:]
    else:
        while BOM.iloc[j,0] > livello_SKU:  
            if (j+1) == len(BOM):
                indice_target = j+1
                df = BOM.iloc[idx:indice_target+1,:]
                break
            j+=1  
            indice_target = j
            df = BOM.iloc[idx:indice_target,:]
    return df


# ### Ottengo moto M, V, X

# In[36]:


SAP_M = partizione(lista_SAP_livelli_1[0],SAP_BOM).reset_index(drop=True)
PLM_M = partizione(lista_PLM_livelli_1[0],PLM_BOM).reset_index(drop=True)
SAP_V = partizione(lista_SAP_livelli_1[1],SAP_BOM).reset_index(drop=True)
PLM_V = partizione(lista_PLM_livelli_1[1],PLM_BOM).reset_index(drop=True)
SAP_X = partizione(lista_SAP_livelli_1[2],SAP_BOM).reset_index(drop=True)
PLM_X = partizione(lista_PLM_livelli_1[2],PLM_BOM).reset_index(drop=True)


# In[37]:


#PLM_M.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Delta BOM/Confronto Python/PLM_M.xlsx',
#            index=False)
#SAP_M.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Delta BOM/Confronto Python/SAP_M.xlsx',
#              index=False)


# ### Tabelle comparative

# In[39]:


def tabella_comparativa (L_SAP,R_PLM):
    SAP_PVT = pd.pivot_table(L_SAP,
                        index='Liv.',
                        values = 'Articolo',
                        aggfunc=['count',pd.Series.nunique])
    SAP_PVT.rename(columns={'Articolo':'SAP','count':'numero righe','nunique':'codici'}
                   ,inplace=True)
    PLM_PVT = pd.pivot_table(R_PLM,
                        index='Liv.',
                        values = 'Articolo',
                        aggfunc=['count',pd.Series.nunique])
    PLM_PVT.rename(columns={'Articolo':'PLM','count':'numero righe','nunique':'codici'}
                   ,inplace=True)
    tabella = pd.concat([SAP_PVT,PLM_PVT],axis=1)
    tabella.fillna(0, inplace=True)
    tabella.sort_index(inplace=True)
    return tabella.astype(int)


# In[ ]:





# ### Layout app

# In[280]:


url_immagine='https://github.com/MarcelloGalimberti/Delta_BOM/blob/main/Ducati-Multistrada-V4-2021-008.jpeg?raw=true'


# In[281]:


st.image(url_immagine)


# In[282]:


# Pubbilcare in Streamlit


# In[283]:


tabella_comparativa(SAP_BOM,PLM_BOM) # fare e visualizzare per moto D, M, V, X


# In[284]:


st.title('MTSV4 MTO Delta BOM')
st.write('Tabella comparativa SAP - PLM')
st.write(tabella_comparativa(SAP_BOM,PLM_BOM))


# ## Analisi moto M

# ---

# #### Test confronto di un sottoalbero di un SKU

# In[41]:


# BOM_L: SAP_M | BOM_R: PLM_M | SKU tra ''
def delta_SKU(SKU,L_BOM,R_BOM):
    df_L=partizione(SKU,L_BOM)
    df_R=partizione(SKU,R_BOM)
    compare = df_L.merge(df_R, how='outer', on='Articolo',
                                   indicator=True,
                                  left_index=False,right_index=False,
                                  suffixes = ('_SAP', '_PLM'))
    compare.rename(columns={'_merge':'Esito check codice'}, inplace=True)
    compare.fillna('',inplace=True)
    compare['Esito check codice'].replace(['both','right_only','left_only'],['Check ok','Non in SAP','Non in PLM'],
                                inplace=True)
    return compare#, df_L, df_R


# In[42]:


# test sul motore


# In[43]:


#codice = '340P9981A' # poi salva file con questo nome


# In[44]:


#compare = delta_SKU(codice,SAP_M,PLM_M) # SKU per test 340P9981A STGR FORCELLA 1706 S | motore 00292141D


# In[45]:


# salva file con nome codice
#compare.to_excel(f'/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Delta BOM/Confronto Python/{codice}.xlsx',
#               index=False)


# ---

# ### Comparazione livelli 2

# In[46]:


def compara_livelli_2 (L_BOM,R_BOM): # L:SAP R:PLM
    SAP_codici_liv_2 = L_BOM[L_BOM['Liv.'] == 2]
    SAP_codici_liv_2.drop(columns=['Liv.','Qty'], inplace=True)
    SAP_codici_liv_2.drop_duplicates(inplace=True)
    PLM_codici_liv_2 = R_BOM[R_BOM['Liv.'] == 2]
    PLM_codici_liv_2.drop(columns=['Liv.','Qty'], inplace=True)
    PLM_codici_liv_2.drop_duplicates(inplace=True)
    comparelev_2 = SAP_codici_liv_2.merge(PLM_codici_liv_2, how='outer', on='Articolo',
                                   indicator=True,
                                  left_index=False,right_index=False,
                                  suffixes = ('_SAP', '_PLM'))
    comparelev_2.rename(columns={'_merge':'Esito check codice'}, inplace=True)
    comparelev_2['Esito check codice'].replace(['both','right_only','left_only'],['Check ok','Non in SAP','Non in PLM'],
                                inplace=True)
    return comparelev_2 # capire se si può togliere


# In[47]:


# esito per codice


# In[48]:


comparelev_2 = compara_livelli_2 (SAP_M,PLM_M)


# In[49]:


comparelev_2


# #### Diagramma di Venn

# In[51]:


lista_Venn = list(comparelev_2['Esito check codice'].value_counts())


# In[52]:


comparelev_2['Esito check codice'].value_counts()


# In[53]:


plt.figure(figsize=(12,6))
venn2(subsets =
     (lista_Venn[1],lista_Venn[2],lista_Venn[0]),
     set_labels=('SAP liv.2','PLM liv.2'),
     alpha=0.5,set_colors=('red', 'yellow'))


# In[54]:


df_confronto_lev2 = comparelev_2[comparelev_2['Esito check codice'] != 'Check ok']


# In[56]:


# ACTION ITEM REGISTER XLS -> poi da scaricare in Streamlit


# In[57]:


df_confronto_lev2


# In[58]:


df_confronto_lev2[['Azione','Resposabile','Due date','Satus']] = ""
#df[["newcol1","newcol2","newcol3"]] = None


# In[59]:


#df_confronto_lev2.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Delta BOM/Confronto Python/AIR.xlsx',
#                          index=False)


# ---

# ### Livelli 2 da controllare con grafo e livelli 2 ok - moto M -

# #### Funzione che individua i livelli 2 senza figli da una BOM

# In[63]:


# Per una BOM mette true ai livelli 2 senza figli


# In[64]:


def livelli_2_depht (BOM):
    W_BOM = BOM.copy()
    W_BOM['Liv.2 senza figli']=False
    for i in range (len(W_BOM)):
        if (i+1 == len(W_BOM) and W_BOM.loc[i,'Liv.']==2):
            W_BOM.loc[i,'Liv.2 senza figli']=True
            break
        elif ((W_BOM.loc[i,'Liv.']==2) and (W_BOM.loc[i+1,'Liv.']==2)):
            W_BOM.loc[i,'Liv.2 senza figli']=True
        else:
            W_BOM.loc[i,'Liv.2 senza figli']=False
    return W_BOM


# In[65]:


def analisi_liv_2(L_BOM,R_BOM): # SAP_M , PLM_M
    comparelev_2 = compara_livelli_2 (L_BOM,R_BOM)
    livelli_2_check_ok = comparelev_2[comparelev_2['Esito check codice'] == 'Check ok']
    articoli_liv2_senza_figli_SAP = livelli_2_depht(L_BOM)
    articoli_liv2_senza_figli_PLM = livelli_2_depht(R_BOM)
    articoli_liv2_senza_figli_SAP=articoli_liv2_senza_figli_SAP[articoli_liv2_senza_figli_SAP['Liv.']==2]
    articoli_liv2_senza_figli_PLM=articoli_liv2_senza_figli_PLM[articoli_liv2_senza_figli_PLM['Liv.']==2]
    livelli_2_per_grafo = livelli_2_check_ok.merge(articoli_liv2_senza_figli_SAP, how='left', on='Articolo',
                                   #indicator=True,
                                  left_index=False,right_index=False)
    livelli_2_per_grafo.rename(columns={'Liv.2 senza figli':'SAP'},inplace=True)
    livelli_2_per_grafo_completo = livelli_2_per_grafo.merge(articoli_liv2_senza_figli_PLM, how='left', on='Articolo',
                                   #indicator=True,
                                  left_index=False,right_index=False)
    livelli_2_per_grafo_completo.rename(columns={'Liv.2 senza figli':'PLM'},inplace=True)
    livelli_2_per_grafo_completo.drop(columns=['Esito check codice','Liv._x','Qty_x','Liv._y','Qty_y'],
                                 inplace=True)
    livelli_2_per_grafo_completo.drop_duplicates(inplace=True)
    livelli_2_per_grafo_completo['Livello 2 ok']=livelli_2_per_grafo_completo['SAP'] & livelli_2_per_grafo_completo['PLM']
    return livelli_2_per_grafo_completo


# In[126]:


analisi_livelli_2 = analisi_liv_2(SAP_M,PLM_M)


# In[127]:


# Livelli due esistenti in entrambe le BOM e che non hanno figli
df_livelli_2_ok = analisi_livelli_2[analisi_livelli_2['Livello 2 ok']==True]


# In[128]:


#  Livelli due esistenti in entrambe le BOM e che hanno figli, quindi serve analisi per livelli più profondi
df_livelli_2_ko_per_grafo = analisi_livelli_2[analisi_livelli_2['Livello 2 ok']==False]


# In[129]:


df_livelli_2_ok # fare report


# In[130]:


df_livelli_2_ko_per_grafo.drop(columns='Livello 2 ok',inplace=True)


# In[132]:


df_livelli_2_ko_per_grafo # per estrazione alberi e confronto (almeno uno dei due ha più di un livello)


# In[133]:


len(df_livelli_2_ko_per_grafo)


# In[144]:


df_livelli_2_ko_per_grafo['Grafo']=df_livelli_2_ko_per_grafo['SAP'] ^ df_livelli_2_ko_per_grafo['PLM']


# In[145]:


per_grafi_accoppiati = df_livelli_2_ko_per_grafo[df_livelli_2_ko_per_grafo['Grafo']==False]


# In[146]:





# ---

# In[ ]:





# ### Grafo

# In[82]:


def albero (SKU,BOM):
    BOM.drop(columns='Qty')
    albero_per_grafo = partizione(SKU,BOM)
    livello_minimo = albero_per_grafo['Liv.'].min()
    albero_per_grafo['Liv.']=albero_per_grafo['Liv.']-livello_minimo
    albero_per_grafo.reset_index(drop=True, inplace=True)
    return albero_per_grafo


# In[88]:


# in ingresso l'output di albero
def grafo(tree):
    tree.drop(columns='Qty')#,inplace=True)
    bom = tree['Liv.'].squeeze()
    source = []
    target = []
    indice_source = []
    indice_target = []
    trova = 0
    for i in range(len(bom)):
    # mettere clausula di uscita
        if i+1 >= len(bom):
            break
        source.append(bom[i+1])
        indice_source.append(i+1)
        if bom[i+1] < bom [i]:
            trova=0
            trova = bom [i+1]-1
            index_list = []
            index_list = [index for index in range(len(target)) if target[index] == trova]
            target.append(target[index_list[-1]])
            indice_target.append(indice_target[index_list[-1]])
        if bom[i+1] > bom[i]:
            target.append(bom[i])
            indice_target.append(i)
        if bom[i+1] == bom [i]:
            #se successivo = sè allora T = liv[i-1]
            target.append(target[i-1])
            indice_target.append(indice_target[i-1])
    data = {'Source': source,
        'Target': target,
        'Indice Source': indice_source,
        'Indice Target': indice_target
       }
    grafo_temp = pd.DataFrame(data)
    grafo_temp['Articolo_Source']=0
    for k in indice_source: 
        grafo_temp['Articolo_Source'][k-1] = tree['Articolo'][k]
    mialista = []
    for j in indice_target:
        mialista.append(j)
    grafo_temp['Articolo_Target']=0
    for k in range(len(indice_target)):
        grafo_temp['Articolo_Target'][k]=tree['Articolo'][mialista[k]]
    return grafo_temp


# ---

# In[149]:


# controlli per grafi con figli: isomorfismo e lista codici con livelli identica


# In[248]:


per_grafi_accoppiati


# In[238]:


def isomorfi (SKU,L_BOM,R_BOM):
    G_SAP=nx.Graph()
    G_PLM=nx.Graph()
    albero_SAP = albero (SKU,L_BOM)
    albero_PLM = albero (SKU,R_BOM)
    G_SAP=nx.from_pandas_edgelist(grafo(albero(SKU,L_BOM)),
                                      source = 'Articolo_Source',
                                      target = 'Articolo_Target')
    G_PLM=nx.from_pandas_edgelist(grafo(albero(SKU,R_BOM)),
                                      source = 'Articolo_Source',
                                      target = 'Articolo_Target')
    Lista_SAP = list(G_SAP.nodes())
    Lista_PLM = list(G_PLM.nodes())
    Lista_SAP.sort()
    Lista_PLM.sort()
    uguali = (Lista_SAP == Lista_PLM)
    
    confronto_liste(Lista_SAP,Lista_PLM)
    sono_isomorfi = nx.vf2pp_is_isomorphic(G_SAP,G_PLM)
    if (uguali == True) and (sono_isomorfi==True):
        alberi_uguali = True
    else:
        alberi_uguali = False
    return alberi_uguali
    


# In[240]:


isomorfi('340P9981A',SAP_M,PLM_M)


# In[254]:


per_grafi_accoppiati


# In[258]:


for i in range(len(per_grafi_accoppiati)):
    art = per_grafi_accoppiati.iloc[i,0]
    per_grafi_accoppiati.iloc[i,4]=isomorfi(art,SAP_M,PLM_M)


# In[259]:


per_grafi_accoppiati


# In[274]:


disegna_grafi('59027993B',SAP_M,PLM_M)


# In[ ]:





# ### Da qui flussi di controllo: singolo vs grafo, grafo vs grafo

# In[ ]:


# due cicli: uno per singolo vs grafo e uno per grafo vs grafo
# nel singolo vs grafo provare add node e rappresentare due grafi oppure mettere il padre nel titolo


# In[134]:


df_livelli_2_ko_per_grafo # se true non ha figli


# In[168]:


def disegna_grafi (SKU,L_BOM,R_BOM):
    G_SAP=nx.Graph()
    G_PLM=nx.Graph()
    albero_SAP = albero (SKU,L_BOM)
    albero_PLM = albero (SKU,R_BOM)
    if len(albero_SAP)==1:
        G_SAP.add_node(SKU)
    else:
        G_SAP=nx.from_pandas_edgelist(grafo(albero(SKU,L_BOM)),
                                      source = 'Articolo_Source',
                                      target = 'Articolo_Target')
    if len(albero_PLM)==1:
        G_PLM.add_node(SKU)
    else:
        G_PLM=nx.from_pandas_edgelist(grafo(albero(SKU,R_BOM)),
                                      source = 'Articolo_Source',
                                      target = 'Articolo_Target')
    plt.figure(figsize=(12,6))
    plt.subplot(121,title=f'SAP {SKU}')
    nx.draw_kamada_kawai(G_SAP, with_labels=True,node_color="red", node_size=150,font_size=6)
    plt.subplot(122,title=f'PLM {SKU}')
    nx.draw_kamada_kawai(G_PLM, with_labels=True,node_color="grey", node_size=150,font_size=6)


# In[169]:


disegna_grafi ('340P9981A',SAP_M,PLM_M)


# In[234]:


disegna_grafi('82119582BC',SAP_M,PLM_M)


# ### Grafo con Networkx e Pyvis

# In[135]:


df_test_SAP = grafo(albero('360S2391A',SAP_M))
df_test_PLM = grafo(albero('360S2391A',PLM_M))


# In[136]:


#fig = plt.figure()
G_SAP=nx.Graph()
G_SAP=nx.from_pandas_edgelist(df_test_SAP, source = 'Articolo_Source', target = 'Articolo_Target')
nx.draw_kamada_kawai(G_SAP, with_labels=True,node_color="red", node_size=100,font_size=10,
              edge_color='black',font_color='black',width=1)
#fig.set_facecolor("#00000F")


# In[137]:


H_PLM=nx.Graph()
H_PLM=nx.from_pandas_edgelist(df_test_PLM, source = 'Articolo_Source', target = 'Articolo_Target')
nx.draw_spring(H_PLM, with_labels=True,node_color="red", node_size=20,font_size=10)


# In[138]:


plt.figure(figsize=(12,6))
G_SAP=nx.from_pandas_edgelist(df_test_SAP, source = 'Articolo_Source', target = 'Articolo_Target')
plt.subplot(121,title='SAP')
nx.draw_kamada_kawai(G_SAP, with_labels=True,node_color="red", node_size=150,font_size=8)
H_PLM=nx.from_pandas_edgelist(df_test_PLM, source = 'Articolo_Source', target = 'Articolo_Target')
plt.subplot(122,title='PLM')
nx.draw_kamada_kawai(H_PLM, with_labels=True,node_color="grey", node_size=150,font_size=8)


# #### Pyvis

# In[139]:


# forse di possono affiancare con streamlit, ma può essere una perdita di tempo


# In[141]:


net1 = Network(height=600,width=800,bgcolor="#222222",font_color="white",
              notebook=True,cdn_resources='in_line')#,select_menu=True, filter_menu=True)
net1.toggle_hide_edges_on_drag(True)
net1.from_nx(H_PLM)
#net.from_nx(G)
#net.show_buttons(filter_=['physics'])
net1.show('ex1.html')


# In[142]:


net2 = Network(height=600,width=800,bgcolor="#222222",font_color="white",
              notebook=True,cdn_resources='in_line')#,select_menu=True, filter_menu=True)
net2.toggle_hide_edges_on_drag(True)
net2.from_nx(G_SAP)
#net.from_nx(G)
#net.show_buttons(filter_=['physics'])
net2.show('ex2.html')

