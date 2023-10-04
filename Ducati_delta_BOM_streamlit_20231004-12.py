#!/usr/bin/env python
# coding: utf-8

# In[107]:


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
st.set_option('deprecation.showPyplotGlobalUse', False)
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import xlsxwriter
import io
from io import StringIO


# In[4]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))


# ---

# In[5]:


def load_data(url, index=''):
    if index:
        data = pd.read_excel(url, index_col=index)
    else:
        data = pd.read_excel(url)
    return data


# In[6]:


url_SAP = 'https://github.com/MarcelloGalimberti/Delta_BOM/blob/main/SAP%20CS12-D33008424-DB10-06.11.2023-Capg.XLSX?raw=true'
url_PLM = 'https://github.com/MarcelloGalimberti/Delta_BOM/blob/main/MBOM_DB10_STIP_EUR_RED_A%20RAGGI_20231003-Filtrato_isSpare.xlsx?raw=true'


# ---

# ### Importazione file da Streamlit

# In[104]:


#uploaded_file_SAP = st.file_uploader("Carica la distinta SAP")


# In[109]:


uploaded_file = st.file_uploader("Carica la distinta SAP")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe_da_streamlit = pd.read_excel(uploaded_file)
    st.write(dataframe_da_streamlit)


# ## Importo ed elaboro file Cap

# In[7]:


SAP_cap = load_data(url_SAP)


# In[8]:


SAP_cap['Liv.']=SAP_cap['Liv. esplosione'].str[-1]


# In[9]:


SAP = SAP_cap.drop(columns=['Liv. esplosione','Numero posizione','Testo breve oggetto',
                           'Unità misura base','Tipo pos.','Testo posizione riga 1',
                           'Testo posizione riga 2','Tipo di materiale','Intra material',
                           'Indice di rilevanza del CCST'])


# In[10]:


SAP = SAP[['Liv.','Numero componenti','Qtà comp. (UMB)','Merce sfusa']]


# In[11]:


SAP.rename(columns={'Numero componenti':'Articolo','Qtà comp. (UMB)':'Qty'},inplace=True)


# In[12]:


SAP = SAP.fillna(0)


# In[13]:


SAP['Merce sfusa'] = SAP['Merce sfusa'].replace('X',1)


# In[14]:


SAP['Liv.']= SAP['Liv.'].astype(int)


# ### Algo per eliminare merce sfusa in file Cap

# In[15]:


SAP_working = SAP.copy()


# In[16]:


SAP_working['Eliminare'] = 0


# In[17]:


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


# In[18]:


SAP_no_merce_sfusa = SAP_working.loc[SAP_working['Eliminare']==0]


# In[19]:


SAP_no_merce_sfusa = SAP_no_merce_sfusa.drop(columns=['Merce sfusa','Eliminare'])


# In[20]:


SAP_no_merce_sfusa.reset_index(drop=True, inplace=True)


# ---

# ## Importo ed elaboro file Siemens

# In[21]:


PLM = load_data(url_PLM)


# In[22]:


#PLM.columns


# In[23]:


PLM_reduced = PLM.drop(columns=['Home', 'Level.1', 'BOM Line', 'Release Statuses',
       'Occurrence Effectivities', 'Effectivity Formula', 'Rev Name',
       'Variant Formula','ID In Context (All Levels)', 'Find No.',
       'Revision Effectivity', 'Unit Of Measure', 'Position Type', 'View Type',
       'Item Type', 'Revision', 'Application Description', 'Text Note 2',
       'Descriptive Connection', 'Is Spare Part', 'Disassemble','Date Released'])


# In[24]:


PLM_reduced.rename(columns={'Level':'Liv.','Item Id':'Articolo','Quantity':'Qty'},inplace=True)


# In[25]:


PLM_reduced.fillna(1,inplace=True)


# ---

# ## Ottengo working file

# In[26]:


# Rinomino entrambi i file
PLM_BOM = PLM_reduced.copy()
SAP_BOM=SAP_no_merce_sfusa.copy()


# In[27]:


# salva in xlsx
#PLM_BOM.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Delta BOM/Confronto Python/PLM_BOM.xlsx',
#                index=False)
#SAP_BOM.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Delta BOM/Confronto Python/SAP_BOM.xlsx',
#                index=False)


# ## Statistiche BOMs e pubblicazione Streamlit

# ### Lista livelli 1  - serve per sottoalberi M, V, X

# In[28]:


SAP_livelli_1 = SAP_BOM[SAP_BOM['Liv.'] == 1]
lista_SAP_livelli_1 = SAP_livelli_1.Articolo.to_list()


# In[29]:


PLM_livelli_1 = PLM_BOM[PLM_BOM['Liv.'] == 1]
lista_PLM_livelli_1 = PLM_livelli_1.Articolo.to_list()


# In[30]:


#lista_SAP_livelli_1


# In[31]:


#lista_PLM_livelli_1


# ### Script per estrarre il sottoalbero delle BOM a partire da un SKU

# In[32]:


# Inserire SKU e BOM nella funzione partizione


# In[33]:


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

# In[34]:


SAP_M = partizione(lista_SAP_livelli_1[0],SAP_BOM).reset_index(drop=True)
PLM_M = partizione(lista_PLM_livelli_1[0],PLM_BOM).reset_index(drop=True)
SAP_V = partizione(lista_SAP_livelli_1[1],SAP_BOM).reset_index(drop=True)
PLM_V = partizione(lista_PLM_livelli_1[1],PLM_BOM).reset_index(drop=True)
SAP_X = partizione(lista_SAP_livelli_1[2],SAP_BOM).reset_index(drop=True)
PLM_X = partizione(lista_PLM_livelli_1[2],PLM_BOM).reset_index(drop=True)


# In[35]:


#PLM_M.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Delta BOM/Confronto Python/PLM_M.xlsx',
#            index=False)
#SAP_M.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Delta BOM/Confronto Python/SAP_M.xlsx',
#              index=False)


# ### Tabelle comparative

# In[36]:


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

# In[37]:


url_immagine='https://github.com/MarcelloGalimberti/Delta_BOM/blob/main/Ducati-Multistrada-V4-2021-008.jpeg?raw=true'


# In[38]:


st.title('MTSV4 MTO Delta BOM')
st.image(url_immagine)


# In[39]:


tabella = tabella_comparativa(SAP_BOM,PLM_BOM) # fare e visualizzare per moto D, M, V, X


# In[40]:


tabella.columns = ['_'.join(col).strip() for col in tabella.columns.values] 


# In[41]:


st.write('Tabella comparativa SAP - PLM')
st.write(tabella)


# ## Analisi moto M

# ---

# #### Test confronto di un sottoalbero di un SKU

# In[42]:


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


# ---

# ### Comparazione livelli 2

# In[43]:


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


# In[44]:


# esito per codice


# In[45]:


comparelev_2 = compara_livelli_2 (SAP_M,PLM_M)


# In[46]:


# comparelev_2


# #### Diagramma di Venn

# In[47]:


lista_Venn = list(comparelev_2['Esito check codice'].value_counts())


# In[102]:


#comparelev_2['Esito check codice'].value_counts()


# In[49]:


fig= plt.figure(figsize=(12,6))
venn2(subsets =
     (lista_Venn[1],lista_Venn[2],lista_Venn[0]),
     set_labels=('SAP liv.2','PLM liv.2'),
     alpha=0.5,set_colors=('red', 'yellow'))


# In[51]:


st.write('Diagramma di Venn per livelli 2')
st.write('Livelli 2 in SAP e non in PLM: ',lista_Venn[1])
st.write('Livelli 2 in comune: ',lista_Venn[0])
st.write('Livelli 2 in PLM e non in SAP: ',lista_Venn[0])


# In[52]:


st.pyplot(fig)


# In[53]:


df_confronto_lev2 = comparelev_2[comparelev_2['Esito check codice'] != 'Check ok']


# In[54]:


# ACTION ITEM REGISTER XLS -> poi da scaricare in Streamlit


# In[55]:


#df_confronto_lev2


# In[110]:


df_confronto_lev2[['Azione','Responsabile','Due date','Satus']] = ""


# In[57]:


st.write('Action Item Register per livelli 2')
st.write(df_confronto_lev2)


# #### Download file csv

# In[103]:


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')   # messo index=False
csv = convert_df(df_confronto_lev2)
st.download_button(
    label="Download Registro delle azioni livelli 2 in CSV",
    data=csv,
    file_name='Registro_Azioni_Livelli_2.csv',
    mime='text/csv',
)


# In[ ]:





# In[ ]:





# ---

# ### Livelli 2 da controllare con grafo e livelli 2 ok - moto M -

# #### Funzione che individua i livelli 2 senza figli da una BOM

# In[61]:


# Per una BOM mette true ai livelli 2 senza figli


# In[62]:


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


# In[63]:


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


# In[64]:


analisi_livelli_2 = analisi_liv_2(SAP_M,PLM_M)


# In[65]:


# Livelli due esistenti in entrambe le BOM e che non hanno figli
df_livelli_2_ok = analisi_livelli_2[analisi_livelli_2['Livello 2 ok']==True]


# In[66]:


#  Livelli due esistenti in entrambe le BOM e che hanno figli, quindi serve analisi per livelli più profondi
df_livelli_2_ko_per_grafo = analisi_livelli_2[analisi_livelli_2['Livello 2 ok']==False]


# In[67]:


# df_livelli_2_ok # fare report


# In[68]:


df_livelli_2_ko_per_grafo.drop(columns='Livello 2 ok',inplace=True)


# In[69]:


# df_livelli_2_ko_per_grafo # per estrazione alberi e confronto (almeno uno dei due ha più di un livello)


# In[70]:


len(df_livelli_2_ko_per_grafo)


# In[71]:


df_livelli_2_ko_per_grafo['Grafo']=df_livelli_2_ko_per_grafo['SAP'] ^ df_livelli_2_ko_per_grafo['PLM']


# In[72]:


per_grafi_accoppiati = df_livelli_2_ko_per_grafo[df_livelli_2_ko_per_grafo['Grafo']==False]


# In[73]:


per_grafi_accoppiati['Isomorfi']=0


# ---

# ### Grafo

# In[74]:


def albero (SKU,BOM):
    BOM.drop(columns='Qty')
    albero_per_grafo = partizione(SKU,BOM)
    livello_minimo = albero_per_grafo['Liv.'].min()
    albero_per_grafo['Liv.']=albero_per_grafo['Liv.']-livello_minimo
    albero_per_grafo.reset_index(drop=True, inplace=True)
    return albero_per_grafo


# In[75]:


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

# In[76]:


# controlli per grafi con figli: isomorfismo e lista codici con livelli identica


# In[77]:


# per_grafi_accoppiati


# In[78]:


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
    sono_isomorfi = nx.vf2pp_is_isomorphic(G_SAP,G_PLM)
    if (uguali == True) and (sono_isomorfi==True):
        alberi_uguali = True
    else:
        alberi_uguali = False
    return alberi_uguali
    


# In[79]:


for i in range(len(per_grafi_accoppiati)):
    art = per_grafi_accoppiati.iloc[i,0]
    per_grafi_accoppiati.iloc[i,4]=isomorfi(art,SAP_M,PLM_M)


# ### Da qui flussi di controllo: singolo vs grafo, grafo vs grafo

# In[80]:


# due cicli: uno per singolo vs grafo e uno per grafo vs grafo
# nel singolo vs grafo provare add node e rappresentare due grafi oppure mettere il padre nel titolo


# In[81]:


# df_livelli_2_ko_per_grafo # se true non ha figli


# In[82]:


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
    #plt.show()


# In[83]:


fig_a = disegna_grafi ('340P9981A',SAP_M,PLM_M)


# In[84]:


#plt.show(fig_a)


# In[85]:


disegna_grafi('82119582BC',SAP_M,PLM_M)


# In[86]:


st.pyplot(fig_a)


# In[ ]:




