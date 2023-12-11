# Importa librerie
import streamlit as st
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import networkx as nx
#from pyvis.network import Network
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from PIL import Image # serve per l'immagine?
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
st.set_option('deprecation.showPyplotGlobalUse', False)
from io import BytesIO
#from pyxlsb import open_workbook as open_xlsb
#import xlsxwriter
import io
from io import StringIO

from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

st.set_page_config(layout="wide")

# Layout app
url_immagine='https://github.com/MarcelloGalimberti/Delta_BOM/blob/main/Ducati-Multistrada-V4-2021-008.jpeg?raw=true'
st.title('ASC MTSV4 MTO Delta BOM rev.20231211')
st.image(url_immagine)

# Importazione file da Streamlit

uploaded_SAP = st.file_uploader("Carica la distinta SAP")
if not uploaded_SAP:
    st.stop()
SAP_cap = pd.read_excel(uploaded_SAP)

st.write(SAP_cap)

uploaded_PLM = st.file_uploader("Carica la distinta PLM")
if not uploaded_PLM:
    #st.warning('Please input a name.')
    # Can be used wherever a "file-like" object is accepted:
    st.stop()
PLM = pd.read_excel(uploaded_PLM)

st.write(PLM)

# Elaboro file Cap

def importa_cap (df):
    df['Liv.']=df['Liv. esplosione'].str.replace('.','')
    df = df[['Liv.','Materiale','Qtà comp. (UMC)','MerceSfusa (BOM)','Ril.Tecn.','Ril.Prod.','Ril.Ric.']]
    df.rename(columns={'Materiale':'Articolo','Qtà comp. (UMC)':'Qty'},inplace=True)
    df = df.fillna(0)
    df['Liv.']= df['Liv.'].astype(int)
    
    # step 1: eliminare tutti i figli con padre livello 2 merce sfusa
    # tiene anche tutti i figli di padri non merce sfusa
    df['Eliminare'] = 0
    for i in range(len(df)):
        if i == len(df):
            break
        if (df.loc[i,'MerceSfusa (BOM)'] == 'Sì' and df.loc[i,'Liv.']== 2):
            livello_padre = df.loc[i,'Liv.']
            # questo tiene anche 43218027X
            if ((df.loc[i,'Ril.Tecn.']==False) and (df.loc[i,'Ril.Prod.']==False) and (df.loc[i,'Ril.Ric.']==False)):
               df.loc[i,'Eliminare']=0
            else:
               df.loc[i,'Eliminare'] = 1
            #
            j = i
            if (j+1) == len(df):
                break
            while df.loc[j+1,'Liv.']>livello_padre:
                df.at[j+1,'Eliminare']=1
                j+=1
                if (j+1) == len(df):
                    break  
    df = df.loc[df['Eliminare']==0]

    # step 2: eliminare il resto in base a rilevanza tecnica
    #df = df.loc[(df['MerceSfusa (BOM)']=='No') | ((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì'))] # questo filtra anche 43218027X
    df = df.loc[(df['MerceSfusa (BOM)']=='No') | ((df['Ril.Prod.']==True)&(df['MerceSfusa (BOM)']=='No')) |((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì')) | (((df['MerceSfusa (BOM)']=='Sì')&(df['Ril.Tecn.']==False)&(df['Ril.Prod.']==False)&(df['Ril.Ric.']==False))==True)]
    #df = df.loc[(df['MerceSfusa (BOM)']=='No') | ((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì')) | (((df['MerceSfusa (BOM)']=='Sì')&(df['Ril.Tecn.']==False)&(df['Ril.Prod.']==False)&(df['Ril.Ric.']==False))==True)]
    df = df.drop(columns=['MerceSfusa (BOM)','Ril.Tecn.','Eliminare','Ril.Prod.','Ril.Ric.'])
    df.reset_index(drop=True, inplace=True)
    return df

# serve per associare descrizione e gruppo tecnico nelle tabelle AIR
def importa_cap_con_descrizone (df):
    df['Liv.']=df['Liv. esplosione'].str.replace('.','')
    df = df[['Liv.','Materiale','Qtà comp. (UMC)','MerceSfusa (BOM)','Ril.Tecn.','Testo breve oggetto','Gruppo Tecnico','Ril.Prod.','Ril.Ric.']]
    df.rename(columns={'Materiale':'Articolo','Qtà comp. (UMC)':'Qty'},inplace=True)
    df = df.fillna(0)
    df['Liv.']= df['Liv.'].astype(int)
    
    # step 1: eliminare tutti i figli con padre livello 2 merce sfusa
    df['Eliminare'] = 0
    for i in range(len(df)):
        if i == len(df):
            break
        if (df.loc[i,'MerceSfusa (BOM)'] == 'Sì' and df.loc[i,'Liv.']== 2):
            livello_padre = df.loc[i,'Liv.']
            #
            if ((df.loc[i,'Ril.Tecn.']==False) and (df.loc[i,'Ril.Prod.']==False) and (df.loc[i,'Ril.Ric.']==False)):
               df.loc[i,'Eliminare']=0
            else:
               df.loc[i,'Eliminare'] = 1
            #
            j = i
            if (j+1) == len(df):
                break
            while df.loc[j+1,'Liv.']>livello_padre:
                df.at[j+1,'Eliminare']=1
                j+=1
                if (j+1) == len(df):
                    break  
    df = df.loc[df['Eliminare']==0] 
   
    # step 2: eliminare il resto in base a rilevanza tecnica
    #df = df.loc[(df['MerceSfusa (BOM)']=='No') | ((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì')) | (((df['MerceSfusa (BOM)']=='Sì')&(df['Ril.Tecn.']==False)&(df['Ril.Prod.']==False)&(df['Ril.Ric.']==False))==True)]
    df = df.loc[(df['MerceSfusa (BOM)']=='No') | ((df['Ril.Prod.']==True)&(df['MerceSfusa (BOM)']=='No')) |((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì')) | (((df['MerceSfusa (BOM)']=='Sì')&(df['Ril.Tecn.']==False)&(df['Ril.Prod.']==False)&(df['Ril.Ric.']==False))==True)]
    #df = df.loc[(df['MerceSfusa (BOM)']=='No') | ((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì'))]
    #df = df.loc[(df['MerceSfusa (BOM)']=='No') | ((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì')) |
    #            (((df['MerceSfusa (BOM)']=='Sì')&(df['Ril.Tecn.']==False)&(df['Ril.Prod.']==False)&(df['Ril.Ric.']==False))==False)]
    
    df = df.drop(columns=['MerceSfusa (BOM)','Ril.Tecn.','Eliminare','Ril.Prod.','Ril.Ric.']) 
    df.reset_index(drop=True, inplace=True) 
    return df


# Elaboro file Siemens | Aggiornare con Layout Massimo

def importa_plm (df):
    df=df[['Level','Item Id','Quantity','Is Spare Part']]
    df = df[df['Is Spare Part'] != 'X']
    df.drop(columns= ['Is Spare Part'], inplace = True)
    df.rename(columns={'Level':'Liv.','Item Id':'Articolo','Quantity':'Qty'},inplace=True)
    df.fillna(1,inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# Ottengo working file

PLM_BOM = importa_plm(PLM)
SAP_BOM=importa_cap(SAP_cap)

#st.write('SAP_BOM')
#st.dataframe(SAP_BOM) #340T9971A


# Lista livelli 1  - serve per sottoalberi M, V, X
SAP_livelli_1 = SAP_BOM[SAP_BOM['Liv.'] == 1]
lista_SAP_livelli_1 = SAP_livelli_1.Articolo.to_list()


PLM_livelli_1 = PLM_BOM[PLM_BOM['Liv.'] == 1]
lista_PLM_livelli_1 = PLM_livelli_1.Articolo.to_list()


# Script per estrarre il sottoalbero delle BOM a partire da un SKU
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


# Processo motore
# 20231205: se non c'è il motore il sottoprocesso deve essere saltato
# Estrarre albero motore (codici che iniziano con...)
# fa 3 slice: prima motore, motore, dopo motore
# Aggiungere albero modificato a file PLM (deve essere inserito in moto M)
# Eliminare caratteristica

# trovo riga che inizia con 0029

indice_motore = 0
for i in range (len (PLM_BOM)):
    if PLM_BOM.loc[i,'Articolo'].startswith('0029'):
        indice_motore = i


if indice_motore > 0:
    # trovo codice motore
    codice_motore = PLM_BOM.loc[indice_motore,'Articolo']

    # estraggo albero motore

    albero_motore = partizione(codice_motore,PLM_BOM)
    indice_inizio = indice_motore
    indice_fine = indice_motore + len(albero_motore)

    # elimino livelli 3 (oppure Item Type == Engine Functional Group) e faccio salire di 1 tutti gli altri tranne il livello 2
    albero_motore['eliminare'] = False
    for i in range (len (albero_motore)):
        if albero_motore.loc[i+indice_inizio,'Articolo'] == codice_motore:
            pass
        elif albero_motore.loc[i+indice_inizio,'Liv.'] == 3:
            albero_motore.loc[i+indice_inizio,'eliminare'] = True
        else:
            albero_motore.loc[i+indice_inizio,'Liv.'] = albero_motore.loc[i+indice_inizio,'Liv.']-1

    albero_motore_processato = albero_motore[albero_motore['eliminare']==False]
    albero_motore_processato.drop(columns=['eliminare'], inplace=True)
    albero_motore_processato.reset_index(inplace=True, drop=True)

    # slice PLM_BOM
    slice_1 = PLM_BOM.iloc[0:indice_inizio,]
    slice_3 = PLM_BOM.iloc[indice_fine:,]

    #st.write('Slice 1',slice_1)
    #st.write('Albero motore', albero_motore_processato)
    #st.write('Slice 3',slice_3)


    # aggiungere albero motore processato
    # questa plm bom valida solo se c'è motore, altrimenti 
    PLM_BOM = pd.concat([slice_1,albero_motore_processato,slice_3], ignore_index=True) # questo non va bene, perchè accodato in fondo è su X

#st.write('PLM BOM dopo if', PLM_BOM)

#def convert_df(df):
#    return df.to_csv(index=False).encode('utf-8')   # messo index=False
#csv = convert_df(PLM_BOM)
#st.download_button(
#    label="Download PLM_BOM new",
#    data=csv,
#    file_name='PLM_BOM new.csv',
#    mime='text/csv',
#)


# Ottengo moto M, V, X
SAP_M = partizione(lista_SAP_livelli_1[0],SAP_BOM).reset_index(drop=True)
PLM_M = partizione(lista_PLM_livelli_1[0],PLM_BOM).reset_index(drop=True)
SAP_V = partizione(lista_SAP_livelli_1[1],SAP_BOM).reset_index(drop=True)
PLM_V = partizione(lista_PLM_livelli_1[1],PLM_BOM).reset_index(drop=True)
SAP_X = partizione(lista_SAP_livelli_1[2],SAP_BOM).reset_index(drop=True)
PLM_X = partizione(lista_PLM_livelli_1[2],PLM_BOM).reset_index(drop=True)



SAP_BOM_con_descrizione = importa_cap_con_descrizone(SAP_cap)

SAP_M_descrizione = partizione(lista_SAP_livelli_1[0],SAP_BOM_con_descrizione).reset_index(drop=True) ####
SAP_V_descrizione = partizione(lista_SAP_livelli_1[1],SAP_BOM_con_descrizione).reset_index(drop=True)
SAP_X_descrizione = partizione(lista_SAP_livelli_1[2],SAP_BOM_con_descrizione).reset_index(drop=True)



# Tabelle comparative
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

tabella = tabella_comparativa(SAP_BOM,PLM_BOM) # fare e visualizzare per moto D, M, V, X

tabella.columns = ['_'.join(col).strip() for col in tabella.columns.values] 

st.header('Tabella comparativa SAP - PLM livelli M, V, X', divider = 'red')

col_a, col_b = st.columns([1,1])

with col_a:
    st.subheader('Tabella', divider = 'red')
    st.write(tabella)

with col_b:
    st.subheader('Descrizione', divider = 'red')
    st.write('Per ogni livello trovato in distinta:')
    st.write('**numero righe_SAP/PLM:** conteggio di codici, anche ripetuti, che compaiono in BOM')
    st.write('**codici_SAP/PLM:** conteggio di codici univoci che compaiono in BOM')   

# Scegliere il livello da analizzare (M,V,X) in streamlit
st.header('Selezionare livello da analizzare', divider = 'red')
livello_1 = st.radio ('Scegli il tipo di distinta livello 1', ['M','V','X'], index=None)
if not livello_1:
    st.stop()

if livello_1 == 'M':
    L_BOM_input = SAP_M
    R_BOM_input = PLM_M
elif livello_1 == 'V':
    L_BOM_input = SAP_V
    R_BOM_input = PLM_V
else:
    L_BOM_input = SAP_X
    R_BOM_input = PLM_X

st.header(f'Analisi moto {livello_1}', divider = 'red')

# Test confronto di un sottoalbero di un SKU
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

# Comparazione livelli 2
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

comparelev_2 = compara_livelli_2 (L_BOM_input,R_BOM_input)
comparelev_2.sort_values(by='Esito check codice', inplace=True, ascending=False) # Ok 348 [0]| Non in SAP 9 [2]| Non in PLM 22 [1]

# ordinare per esito check codice?? check ok

#st.write('compare lev 2',comparelev_2)

check_ok = len(comparelev_2[comparelev_2['Esito check codice'] == 'Check ok'])
#st.write('check ok', check_ok)
non_in_sap = len(comparelev_2[comparelev_2['Esito check codice'] == 'Non in SAP'])
#st.write('non in sap', non_in_sap)
non_in_plm = len(comparelev_2[comparelev_2['Esito check codice'] == 'Non in PLM'])
#st.write('non in plm', non_in_plm)

kpi_liv2 = (non_in_plm + non_in_sap)/(check_ok+non_in_sap+non_in_plm)*100

lista_Venn=[check_ok,non_in_plm, non_in_sap]

# Diagramma di Venn
#lista_Venn = list(comparelev_2['Esito check codice'].value_counts(sort=False))

#st.write('lista venn', lista_Venn)
#st.write('len compare lev 2', len(comparelev_2))

fig= plt.figure(figsize=(12,6))
venn2(subsets =
     (lista_Venn[1],lista_Venn[2],lista_Venn[0]),
     set_labels=('SAP liv.2','PLM liv.2'),
     alpha=0.5,set_colors=('red', 'yellow'))

df_confronto_lev2 = comparelev_2[comparelev_2['Esito check codice'] != 'Check ok']
df_confronto_lev2[['Azione','Responsabile','Due date','Status']] = ""

# Aggiunto algo per descrizione e gruppo tecnico

if livello_1 == 'M':
    df_SAP_descrizione = SAP_M_descrizione[['Articolo','Testo breve oggetto','Gruppo Tecnico']]
elif livello_1 == 'V':
    df_SAP_descrizione = SAP_V_descrizione[['Articolo','Testo breve oggetto','Gruppo Tecnico']]
else:
    df_SAP_descrizione = SAP_X_descrizione[['Articolo','Testo breve oggetto','Gruppo Tecnico']]


#df_SAP_descrizione = SAP_cap[['Materiale','Testo breve oggetto','Gruppo Tecnico']]
df_PLM_descrizione = PLM[['Item Id','Rev Name']]

codici_non_in_PLM = df_confronto_lev2[df_confronto_lev2['Esito check codice']=='Non in PLM']
AIR_SAP = codici_non_in_PLM.merge(df_SAP_descrizione,how='left',left_on='Articolo', right_on='Articolo')
#AIR_SAP = AIR_SAP.drop(columns=['Articolo'])
AIR_SAP = AIR_SAP[['Articolo','Testo breve oggetto','Gruppo Tecnico','Esito check codice','Azione','Responsabile','Due date','Status']]
AIR_SAP.rename(columns={'Testo breve oggetto': 'Descrizione'}, inplace=True)
#st.write(AIR_SAP)

codici_non_in_SAP = df_confronto_lev2[df_confronto_lev2['Esito check codice']=='Non in SAP']
AIR_PLM = codici_non_in_SAP.merge(df_PLM_descrizione,how = 'left',left_on='Articolo', right_on='Item Id')
AIR_PLM = AIR_PLM.drop(columns=['Item Id'])
AIR_PLM = AIR_PLM[['Articolo','Rev Name','Esito check codice','Azione','Responsabile','Due date','Status']]
AIR_PLM.rename(columns={'Rev Name':'Descrizione'},inplace=True)
#st.write(AIR_PLM)

df_confronto_lev2_descrizione = pd.concat([AIR_SAP,AIR_PLM],ignore_index=True)
df_confronto_lev2_descrizione.drop_duplicates(inplace=True)
df_confronto_lev2_descrizione.reset_index(drop=True, inplace=True)

# st.write(df_confronto_lev2_descrizione)
# Mettere output in due colonne

col1, col2 = st.columns([2,1])

with col1:
    st.subheader('Action Item livelli 2', divider = 'red')
    st.write(df_confronto_lev2_descrizione)
    st.markdown('**KPI 1: percentuale livelli 2 ok: :green[{:0.1f}%]**'.format(100-kpi_liv2)) #0,. per separatore miglialia

with col2:
    st.subheader('Venn livelli 2', divider = 'red')
    st.pyplot(fig)
    st.write('Livelli 2 in SAP e **non in PLM**: ',lista_Venn[1]) ### modificato
    st.write('Livelli 2 in comune: ',lista_Venn[0]) ### modificato
    st.write('Livelli 2 in PLM e **non in SAP**: ',lista_Venn[2]) ### modificato


# Download file csv

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')   # messo index=False
csv = convert_df(df_confronto_lev2_descrizione)
st.download_button(
    label="Download Registro delle azioni livelli 2 in CSV",
    data=csv,
    file_name='Registro_Azioni_Livelli_2.csv',
    mime='text/csv',
)

##### Confronto qty livelli 2
# Aggiunto algo per qty

if livello_1 == 'M':
    df_SAP_des_qty = SAP_M_descrizione[['Liv.','Articolo','Qty','Testo breve oggetto','Gruppo Tecnico']]
elif livello_1 == 'V':
    df_SAP_des_qty = SAP_V_descrizione[['Liv.','Articolo','Qty','Testo breve oggetto','Gruppo Tecnico']]
else:
    df_SAP_des_qty = SAP_X_descrizione[['Liv.','Articolo','Qty','Testo breve oggetto','Gruppo Tecnico']]

df_PLM_2 = R_BOM_input[R_BOM_input['Liv.']==2]
pvt_plm_2 = pd.pivot_table(df_PLM_2, index=['Articolo'],values = ['Qty'], aggfunc = 'sum')
pvt_plm_2.rename(columns={'Qty':'Qty_PLM'}, inplace=True)

df_SAP_2=df_SAP_des_qty[df_SAP_des_qty['Liv.']==2]

pvt_sap_2 = pd.pivot_table(df_SAP_2, index=['Articolo'],values = ['Qty'], aggfunc = 'sum')
pvt_sap_2.rename(columns={'Qty':'Qty_SAP'}, inplace=True)

# Confronto quantità totali
delta_qty_liv_2_1 = pvt_sap_2.merge(pvt_plm_2,how='outer',left_on='Articolo',right_on='Articolo',indicator=True,
                                  left_index=False,right_index=False,
                                  suffixes = ('_SAP', '_PLM'))
delta_qty_liv_2_1['Delta_qty']=delta_qty_liv_2_1.Qty_SAP-delta_qty_liv_2_1.Qty_PLM
delta_qty_table = delta_qty_liv_2_1[(delta_qty_liv_2_1['_merge']=='both') & (delta_qty_liv_2_1['Delta_qty']!=0)]
delta_qty_table.reset_index(inplace=True)
delta_qty_table.drop(columns=['_merge'], inplace=True)

st.subheader('Delta quantità codici livello 2',divider='red')
st.dataframe(delta_qty_table)

# Livelli 2 da controllare con grafo e livelli 2 ok - moto M -
# Funzione che individua i livelli 2 senza figli da una BOM
# Per una BOM mette true ai livelli 2 senza figli

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

analisi_livelli_2 = analisi_liv_2(L_BOM_input,R_BOM_input)

# Livelli due esistenti in entrambe le BOM e che non hanno figli
df_livelli_2_ok = analisi_livelli_2[analisi_livelli_2['Livello 2 ok']==True]

# Report di df_livelli_2_ok in Stremlit

st.header(f'Analisi moto {livello_1} livelli 2 in comune', divider = 'red')
st.write(f'Su un totale di {lista_Venn[0]} codici comuni, {len(df_livelli_2_ok)} non hanno figli.') ## modificato
st.write('**Per questi codici neussuna ulteriore azione è richiesta**') # poi mettere colore
metrica = round(len(df_livelli_2_ok)/len(analisi_livelli_2)*100,1)
#st.markdown('**KPI: percentuale livelli 2 in comune che non necessita azioni :green[{:0.1f}%]**'.format(100-metrica))
#st.metric(label="Percentuale livelli 2 che necessita di azioni", value=100-metrica)
st.markdown('**Percentuale livelli 2 che necessita di azioni: :red[{:0.1f}%]**'.format(100-metrica)) 

##############

#  Livelli due esistenti in entrambe le BOM e che hanno figli, quindi serve analisi per livelli più profondi
df_livelli_2_ko_per_grafo = analisi_livelli_2[analisi_livelli_2['Livello 2 ok']==False]
df_livelli_2_ko_per_grafo.drop(columns='Livello 2 ok',inplace=True)
df_livelli_2_ko_per_grafo['Grafo']=df_livelli_2_ko_per_grafo['SAP'] ^ df_livelli_2_ko_per_grafo['PLM'] # ^ è XOR

# st.write('Blocco', df_livelli_2_ko_per_grafo) # ok, non si pianta qui


# Funzioni per disegnare grafi: albero, grafo, isomorfi_2

# Grafo
#@st.cache_resource()
def albero (SKU,BOM):
    BOM.drop(columns='Qty')
    albero_per_grafo = partizione(SKU,BOM)
    livello_minimo = albero_per_grafo['Liv.'].min()
    albero_per_grafo['Liv.']=albero_per_grafo['Liv.']-livello_minimo
    albero_per_grafo.reset_index(drop=True, inplace=True)
    return albero_per_grafo


# in ingresso l'output di albero
#@st.cache_resource()
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


def isomorfi_2 (SKU,L_BOM,R_BOM):
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

# funzione per isomorfismo e quantità
@st.cache_resource()
def isomorfi_3 (SKU,L_BOM,R_BOM):  # fa anche confronto qty
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
    
    ### quantità ai nodi
    #for i in range (len(albero_PLM)):
    #    articolo = albero_PLM.loc[i,'Articolo']
    #    G_PLM.nodes[articolo]['Qty']=albero_PLM.loc[i,'Qty']
    #for i in range (len(albero_SAP)):
    #    articolo = albero_SAP.loc[i,'Articolo']
    #    G_SAP.nodes[articolo]['Qty']=albero_SAP.loc[i,'Qty']    
    
    
    Lista_SAP = list(G_SAP.nodes())
    Lista_PLM = list(G_PLM.nodes())
    Lista_SAP.sort()
    Lista_PLM.sort()
    uguali = (Lista_SAP == Lista_PLM)
    merge_alberi = albero_SAP.merge(albero_PLM, how = 'left', left_on='Articolo', right_on='Articolo', suffixes=['_SAP','_PLM'])
    stesse_qty = min(merge_alberi['Qty_SAP']==merge_alberi['Qty_PLM'])
    sono_isomorfi = nx.vf2pp_is_isomorphic(G_SAP,G_PLM)
    if (uguali == True) and (sono_isomorfi==True) and (stesse_qty == True):
        alberi_uguali = True
    else:
        alberi_uguali = False
    return alberi_uguali



# Seleziona codice, disegna grafi e fa report

df_livelli_2_ko_per_grafo['Isomorfi']=0

#
df_livelli_2_ko_per_grafo.reset_index(inplace = True, drop=True)

#

#st.write('Blocco_2 livelli 2 ko per grafo', df_livelli_2_ko_per_grafo) #ok, non si pianta qui

for i in range(len(df_livelli_2_ko_per_grafo)):
    art = df_livelli_2_ko_per_grafo.loc[i,'Articolo']     #art = df_livelli_2_ko_per_grafo.iloc[i,0]
    df_livelli_2_ko_per_grafo.loc[i,'Isomorfi']=isomorfi_3(art,L_BOM_input,R_BOM_input)   #isomorfi_3  df_livelli_2_ko_per_grafo.iloc[i,4]=isomorfi_3(art,L_BOM_input,R_BOM_input)

per_grafo = df_livelli_2_ko_per_grafo[df_livelli_2_ko_per_grafo['Isomorfi']==False]

#st.write('Blocco_3', per_grafo) # qui si pianta già


codici_grafo = per_grafo['Articolo'] # serie per selezione codici
numero_codici_grafo = len(codici_grafo)
numero_codici_isomorfi=len(df_livelli_2_ko_per_grafo[df_livelli_2_ko_per_grafo['Isomorfi']==True])

st.header(f'Analisi moto {livello_1} livelli 2 con figli', divider = 'red' )
st.write(f'Codici con alberi isomorfi: {numero_codici_isomorfi} | **non necessitano di analisi**')
st.write(f'Codici da analizzare: {numero_codici_grafo}')
kpi_liv_2_con_figli = numero_codici_grafo/(numero_codici_grafo+numero_codici_isomorfi)*100
st.markdown('**KPI 2: percentuale livelli 2 con figli ok: :green[{:0.1f}%]**'.format(100-kpi_liv_2_con_figli)) #0,. per separatore miglialia

st.subheader('Analisi complessiva Delta BOM livelli 2')


df_delta_complessivo = pd.DataFrame()
for codice in codici_grafo:
    albero_L = albero(codice,L_BOM_input) # SAP_M (ad esempio)
    albero_R = albero(codice,R_BOM_input) # PLM_M (ad esempio)
    df_output_confronto_L = albero_L
    df_output_confronto_L['Liv.']=df_output_confronto_L['Liv.']+2
    df_output_confronto_L.rename(columns= {'Liv.':'Liv. SAP','Articolo':'Articolo SAP','Qty':'Qty SAP'},inplace=True)
    # aggiunge descrizione e gruppo tecnico
    AIR_SAP_sub = df_output_confronto_L.merge(df_SAP_descrizione,how='left',left_on='Articolo SAP', right_on='Articolo')
    #AIR_SAP_sub = AIR_SAP_sub.drop(columns=['Materiale'])
    AIR_SAP_sub = AIR_SAP_sub[['Liv. SAP','Articolo SAP','Testo breve oggetto','Gruppo Tecnico','Qty SAP']]
    AIR_SAP_sub.rename(columns={'Testo breve oggetto':'Descrizione SAP'},inplace=True)
    AIR_SAP_sub.drop_duplicates(inplace=True)
    AIR_SAP_sub.reset_index(drop=True, inplace=True)

    df_output_confronto_R = albero_R
    df_output_confronto_R['Liv.']=df_output_confronto_R['Liv.']+2
    df_output_confronto_R.rename(columns= {'Liv.':'Liv. PLM','Articolo':'Articolo PLM','Qty':'Qty PLM'},inplace = True)
    # aggiunge descrizione
    AIR_PLM_sub = df_output_confronto_R.merge(df_PLM_descrizione,how = 'left',left_on='Articolo PLM', right_on='Item Id')
    AIR_PLM_sub = AIR_PLM_sub.drop(columns='Item Id')
    AIR_PLM_sub = AIR_PLM_sub[['Liv. PLM','Articolo PLM','Rev Name','Qty PLM']]
    AIR_PLM_sub.rename(columns={'Rev Name':'Descrizione PLM'},inplace=True)
    AIR_PLM_sub.drop_duplicates(inplace=True)
    AIR_PLM_sub.reset_index(drop=True, inplace=True)
    df_output_confronto = pd.concat([AIR_SAP_sub,AIR_SAP_sub],axis=1)
    df_delta = AIR_SAP_sub.merge(AIR_PLM_sub,how='outer', left_on='Articolo SAP', right_on='Articolo PLM',
                             indicator=True,
                                  left_index=False,right_index=False)#,
                                  #suffixes = ('_SAP', '_PLM'))
    df_delta.rename(columns={'_merge':'Delta'}, inplace=True)
    df_delta.replace({'left_only':'Solo in SAP','right_only':'Solo in PLM','both':'Comune'},inplace=True)
    df_delta['Delta qty SAP-PLM']=df_delta['Qty SAP']-df_delta['Qty PLM']
    df_delta_complessivo = pd.concat([df_delta_complessivo,df_delta],ignore_index=False)


righe_totali = len(df_delta_complessivo)
righe_ok = len(df_delta_complessivo[df_delta_complessivo['Delta qty SAP-PLM']==0])
st.write('Righe totali: ',righe_totali)
st.write('Righe ok: ', righe_ok)
st.markdown('**KPI 3: percentuale righe ok: :green[{:0.1f}%]**'.format(righe_ok/righe_totali*100)) #0,. per separatore miglialia

st.dataframe(df_delta_complessivo)

csv_3 = convert_df(df_delta_complessivo)
st.download_button(
    label=f"Download confronto complessivo in CSV",
    data=csv_3,
    file_name='delta_complessivo.csv',
    mime='text/csv',
)
#### tabella delta complessiva per livello di distinta


part_number = st.selectbox('Seleziona articolo da analizzare',codici_grafo,index=None )
if not part_number:
    st.stop()


# Sceglie BOM in base al livello moto

albero_L = albero(part_number,L_BOM_input) # SAP_M (ad esempio)
albero_R = albero(part_number,R_BOM_input) # PLM_M (ad esempio)

# Da qui flussi di controllo: singolo vs grafo, grafo vs grafo

# due cicli: uno per singolo vs grafo e uno per grafo vs grafo
# nel singolo vs grafo provare add node e rappresentare due grafi oppure mettere il padre nel titolo

# df_livelli_2_ko_per_grafo # se true non ha figli

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
    nx.draw_kamada_kawai(G_SAP, with_labels=True,node_color="red", node_size=150,font_size=8)
    plt.subplot(122,title=f'PLM {SKU}')
    nx.draw_kamada_kawai(G_PLM, with_labels=True,node_color="grey", node_size=150,font_size=8)
    #plt.show()

df_output_confronto_L = albero_L
df_output_confronto_L['Liv.']=df_output_confronto_L['Liv.']+2
df_output_confronto_L.rename(columns= {'Liv.':'Liv. SAP','Articolo':'Articolo SAP','Qty':'Qty SAP'},inplace=True)
# aggiunge descrizione e gruppo tecnico
AIR_SAP_sub = df_output_confronto_L.merge(df_SAP_descrizione,how='left',left_on='Articolo SAP', right_on='Articolo')
#AIR_SAP_sub = AIR_SAP_sub.drop(columns=['Materiale'])
AIR_SAP_sub = AIR_SAP_sub[['Liv. SAP','Articolo SAP','Testo breve oggetto','Gruppo Tecnico','Qty SAP']]
AIR_SAP_sub.rename(columns={'Testo breve oggetto':'Descrizione SAP'},inplace=True)
AIR_SAP_sub.drop_duplicates(inplace=True)
AIR_SAP_sub.reset_index(drop=True, inplace=True)

#st.write(AIR_SAP_sub)

df_output_confronto_R = albero_R
df_output_confronto_R['Liv.']=df_output_confronto_R['Liv.']+2
df_output_confronto_R.rename(columns= {'Liv.':'Liv. PLM','Articolo':'Articolo PLM','Qty':'Qty PLM'},inplace = True)
# aggiunge descrizione
AIR_PLM_sub = df_output_confronto_R.merge(df_PLM_descrizione,how = 'left',left_on='Articolo PLM', right_on='Item Id')
AIR_PLM_sub = AIR_PLM_sub.drop(columns='Item Id')
AIR_PLM_sub = AIR_PLM_sub[['Liv. PLM','Articolo PLM','Rev Name','Qty PLM']]
AIR_PLM_sub.rename(columns={'Rev Name':'Descrizione PLM'},inplace=True)
AIR_PLM_sub.drop_duplicates(inplace=True)
AIR_PLM_sub.reset_index(drop=True, inplace=True)

#st.write(AIR_PLM_sub)

# Nuovo layout

#st.subheader('Codici SAP', divider = 'red')
#st.write(AIR_SAP_sub)

st.subheader('Grafo SAP | PLM', divider = 'red')
fig_a = disegna_grafi(part_number,L_BOM_input,R_BOM_input)
st.pyplot(fig_a)

#st.subheader('Codici PLM', divider = 'red')
#st.write(AIR_PLM_sub)


df_output_confronto = pd.concat([AIR_SAP_sub,AIR_SAP_sub],axis=1)
df_delta = AIR_SAP_sub.merge(AIR_PLM_sub,how='outer', left_on='Articolo SAP', right_on='Articolo PLM',
                             indicator=True,
                                  left_index=False,right_index=False)#,
                                  #suffixes = ('_SAP', '_PLM'))
df_delta.rename(columns={'_merge':'Delta'}, inplace=True)
df_delta.replace({'left_only':'Solo in SAP','right_only':'Solo in PLM','both':'Comune'},inplace=True)
df_delta['Delta qty SAP-PLM']=df_delta['Qty SAP']-df_delta['Qty PLM']
#df_delta['Delta']=df_delta['Delta'].str.replace()


st.subheader(f'Tabella delta {part_number}', divider='red')
st.write(df_delta)


csv_2 = convert_df(df_delta)
st.download_button(
    label=f"Download confronto {part_number} in CSV",
    data=csv_2,
    file_name=f'{part_number}.csv',
    mime='text/csv',
)