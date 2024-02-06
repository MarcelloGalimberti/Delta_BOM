# Importa librerie

# 2024 02 06 | 
# inserita sezione per il confronto flag livelli 2 da tabella action item register
# inserita logica fuzzy per la ricerca dei  sottogruppi mancanti da sap (da aggiungere nella chiave anche la quantità su suggerimento di Giac)



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
#from fuzzywuzzy import fuzz

from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

st.set_page_config(layout="wide")

tab1, tab2, tab3, tab4 = st.tabs(['Input dati','Action Item liv2', 'Delta qty liv2', 'Other'])

with tab1: #Input dei dati
    # Layout app
    url_immagine='https://github.com/MarcelloGalimberti/Delta_BOM/blob/main/Ducati-Multistrada-V4-2021-008.jpeg?raw=true'
    st.title('Delta BOM D33 vs DAA rev.20240115')
    #st.image(url_immagine)

    # Importazione file da Streamlit

    uploaded_SAP = st.file_uploader("Carica la distinta D33")
    if not uploaded_SAP:
        st.stop()
    SAP_cap = pd.read_excel(uploaded_SAP)

    st.write(SAP_cap[0:2])

    uploaded_PLM = st.file_uploader("Carica la distinta DAA")
    if not uploaded_PLM:
        #st.warning('Please input a name.')
        # Can be used wherever a "file-like" object is accepted:
        st.stop()
    PLM = pd.read_excel(uploaded_PLM)

    st.write(PLM[0:2])

    sku = PLM['Numero componenti'].iloc[0]
    st.sidebar.header(f'SKU: :red[{sku}]')

    anagrafica_sap = SAP_cap[['Materiale','Testo breve oggetto']]
    anagrafica_sap.drop_duplicates(inplace=True)
    anagrafica_plm = PLM[['Numero componenti','Testo breve oggetto']]
    anagrafica_plm.drop_duplicates(inplace=True)

    #st.write(anagrafica_sap)
    #st.write(anagrafica_plm)


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
        df = df.loc[(df['MerceSfusa (BOM)']=='No') | ((df['Ril.Prod.']==True)&(df['MerceSfusa (BOM)']=='No')) |((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì')) | (((df['MerceSfusa (BOM)']=='Sì')&(df['Ril.Tecn.']==False)&(df['Ril.Prod.']==False)&(df['Ril.Ric.']==False))==True)]
        df = df.drop(columns=['MerceSfusa (BOM)','Ril.Tecn.','Eliminare','Ril.Prod.','Ril.Ric.']) 
        df.reset_index(drop=True, inplace=True) 
        return df

    #AB inizio-------------------------
    def sap_raw (df):
        df['Liv.']=df['Liv. esplosione'].str.replace('.','')
        df = df[['Liv.','Materiale','Qtà comp. (UMC)','MerceSfusa (BOM)','Ril.Tecn.','Testo breve oggetto','Gruppo Tecnico','Descr. Gruppo Tecnico','Ril.Prod.','Ril.Ric.','Testo posizione riga 1',
        'Testo posizione riga 2','STGR','Descrizione Sottogruppo','Gruppo appartenenza','Descr. Gruppo Appartenenza']]
        df.rename(columns={'Materiale':'Articolo','Qtà comp. (UMC)':'Qty'},inplace=True)
        return df

    def plm_raw (df):
        df['Liv.']=df['Liv. esplosione'].str.replace('.','')
        df = df[['Liv.','Numero componenti','Qtà comp. (UMC)','Merce sfusa','Ril. progettazione','Testo breve oggetto','Gruppo Tecnico','Descr. Gruppo Tecnico','Rilevante produzione','Cd.parte di ricambio','Testo posizione riga 1',
        'Testo posizione riga 2','STGR','Descrizione Sottogruppo','Gruppo appartenenza','Descr. Gruppo Appartenenza']]
        df.rename(columns={'Numero componenti':'Articolo','Qtà comp. (UMC)':'Qty','Merce sfusa':'MerceSfusa (BOM)','Ril. progettazione':'Ril.Tecn.','Rilevante produzione':'Ril.Prod.','Cd.parte di ricambio':'Ril.Ric.'},
                inplace=True)
        #df = df.fillna(0) eliminato 28/12
        df['Liv.']= df['Liv.'].astype(int)

        df['MerceSfusa (BOM)']=df['MerceSfusa (BOM)'].apply(lambda x: 'Sì' if x == 'X' else 'No' )
        
        df['Ril.Tecn.']=df['Ril.Tecn.'].apply(lambda x: True if x  =='X' else False)
        df['Ril.Prod.']=df['Ril.Prod.'].apply(lambda x: True if x  =='X' else False)
        df['Ril.Ric.']=df['Ril.Ric.'].apply(lambda x: True if x  =='X' else False)
        return df

    #AB fine----------------------------

    def importa_cap_con_descrizone_D33 (df):
        df['Liv.']=df['Liv. esplosione'].str.replace('.','')
        df = df[['Liv.','Materiale','Qtà comp. (UMC)','MerceSfusa (BOM)','Ril.Tecn.','Testo breve oggetto','Gruppo Tecnico','Descr. Gruppo Tecnico','Ril.Prod.','Ril.Ric.','Testo posizione riga 1',
                'Testo posizione riga 2','STGR','Descrizione Sottogruppo','Gruppo appartenenza','Descr. Gruppo Appartenenza']]
        df.rename(columns={'Materiale':'Articolo','Qtà comp. (UMC)':'Qty'},inplace=True)
        #df = df.fillna(0) eliminato 28/12
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
        df = df.loc[(df['MerceSfusa (BOM)']=='No') | ((df['Ril.Prod.']==True)&(df['MerceSfusa (BOM)']=='No')) |((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì')) | (((df['MerceSfusa (BOM)']=='Sì')&(df['Ril.Tecn.']==False)&(df['Ril.Prod.']==False)&(df['Ril.Ric.']==False))==True)] 
        df = df.drop(columns=['MerceSfusa (BOM)','Ril.Tecn.','Eliminare','Ril.Prod.','Ril.Ric.']) 
        df.reset_index(drop=True, inplace=True) 
        return df


    # Elaboro file Siemens | Aggiornare con Layout Massimo

    def importa_plm_con_descrizone_DAA (df):
        df['Liv.']=df['Liv. esplosione'].str.replace('.','')
        df = df[['Liv.','Numero componenti','Qtà comp. (UMC)','Merce sfusa','Ril. progettazione','Testo breve oggetto','Gruppo Tecnico','Descr. Gruppo Tecnico','Rilevante produzione','Cd.parte di ricambio','Testo posizione riga 1',
                'Testo posizione riga 2','STGR','Descrizione Sottogruppo','Gruppo appartenenza','Descr. Gruppo Appartenenza']]

        df.rename(columns={'Numero componenti':'Articolo','Qtà comp. (UMC)':'Qty','Merce sfusa':'MerceSfusa (BOM)','Ril. progettazione':'Ril.Tecn.','Rilevante produzione':'Ril.Prod.','Cd.parte di ricambio':'Ril.Ric.'},
                inplace=True)
        #df = df.fillna(0) eliminato 28/12
        df['Liv.']= df['Liv.'].astype(int)

        df['MerceSfusa (BOM)']=df['MerceSfusa (BOM)'].apply(lambda x: 'Sì' if x == 'X' else 'No' )
        
        df['Ril.Tecn.']=df['Ril.Tecn.'].apply(lambda x: True if x  =='X' else False)
        df['Ril.Prod.']=df['Ril.Prod.'].apply(lambda x: True if x  =='X' else False)
        df['Ril.Ric.']=df['Ril.Ric.'].apply(lambda x: True if x  =='X' else False)

        
        #correzione gruppi teecnici mancanti
        for i in range(len(df)): 
            if (df['Liv.'].iloc[i] >2) and (df['Gruppo Tecnico'].astype(str).iloc[i]=='nan'):
                df['Gruppo Tecnico'].iloc[i] = df['Gruppo Tecnico'].iloc[i-1]
                df['Descr. Gruppo Tecnico'].iloc[i] = df['Descr. Gruppo Tecnico'].iloc[i-1]

        # step 1: eliminare tutti i figli con padre livello 2 merce sfusa
        df['Eliminare'] = 0
        for i in range(len(df)):
            if i == len(df):
                break
            if (df.loc[i,'MerceSfusa (BOM)'] == 'Sì' and df.loc[i,'Liv.']== 2):
                livello_padre = df.loc[i,'Liv.']
                #
                #if ((df.loc[i,'Ril.Tecn.']==False) and (df.loc[i,'Ril.Prod.']==False) and (df.loc[i,'Ril.Ric.']==False)) or ((df.loc[i,'Ril.Prod.']==True) and (df.loc[i,'Ril.Ric.']==True)) :
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

            if (df.loc[i,'MerceSfusa (BOM)'] == 'No' and df.loc[i,'Liv.']== 2) : # eliminazione codici SOLO RICAMBIO
                if ((df.loc[i,'Ril.Tecn.']==True) and (df.loc[i,'Ril.Prod.']==False) and (df.loc[i,'Ril.Ric.']==True)):
                    df.loc[i,'Eliminare'] = 1
                else:
                    df.loc[i,'Eliminare'] = 0

        df = df.loc[df['Eliminare']==0] 
    
        # step 2: eliminare il resto in base a rilevanza tecnica
        df = df.loc[(df['MerceSfusa (BOM)']=='No') | ((df['Ril.Prod.']==True)&(df['MerceSfusa (BOM)']=='No')) |((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì')) | (((df['MerceSfusa (BOM)']=='Sì')&(df['Ril.Tecn.']==False)&(df['Ril.Prod.']==False)&(df['Ril.Ric.']==False))==True)]
        #df = df.loc[(df['MerceSfusa (BOM)']=='No') | ((df['Ril.Prod.']==True)&(df['MerceSfusa (BOM)']=='No')) |((df['Ril.Prod.']==True)&(df['Ril.Ric.']==True)&(df['MerceSfusa (BOM)']=='Sì')) | (((df['MerceSfusa (BOM)']=='Sì')&(df['Ril.Tecn.']==False)&(df['Ril.Prod.']==False)&(df['Ril.Ric.']==False))==True)]
        df = df.drop(columns=['MerceSfusa (BOM)','Ril.Tecn.','Eliminare','Ril.Prod.','Ril.Ric.']) 
        df.reset_index(drop=True, inplace=True) 
        return df



    # Ottengo working file

    # modifica 20231228
    SAP_BOM = importa_cap_con_descrizone_D33(SAP_cap)
    PLM_BOM = importa_plm_con_descrizone_DAA (PLM)

    SAP_raw = sap_raw(SAP_cap)
    PLM_raw = plm_raw(PLM)


    # Lista livelli 1  - serve per sottoalberi M, V, X
    # ok 20231228
    SAP_livelli_1 = SAP_BOM[SAP_BOM['Liv.'] == 1]
    lista_SAP_livelli_1 = SAP_livelli_1.Articolo.to_list()


    PLM_livelli_1 = PLM_BOM[PLM_BOM['Liv.'] == 1]
    lista_PLM_livelli_1 = PLM_livelli_1.Articolo.to_list()


    # Script per estrarre il sottoalbero delle BOM a partire da un SKU
    # ok 20231228
    def partizione (SKU,BOM):#------------------------------------------------------------------------------------------------------------------------------------SEGNALIBRO
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

    # trovo riga che inizia con 0029
    # ok 20231228
    indice_motore = 0
    for i in range (len (PLM_BOM)):
        if PLM_BOM.loc[i,'Articolo'].startswith('0029'):
            indice_motore = i
        # st.write(indice_motore)
        # st.write(PLM_BOM.loc[i,'Articolo'])
        # st.write(PLM_BOM)


    if indice_motore > 0:    # eliminazione righe e assegnazione gt
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

        # assegno i gruppi tecnici al motore
        for i in range(1,len(albero_motore)):
            if albero_motore.eliminare.iloc[i]==True:
                gruppo_tecnico = albero_motore.Articolo.astype(str).iloc[i][:2]+'.'
                des_gruppo = albero_motore['Testo breve oggetto'].iloc[i]
            albero_motore['Gruppo Tecnico'].iloc[i]=gruppo_tecnico
            albero_motore['Descr. Gruppo Tecnico'].iloc[i]=des_gruppo
                
        albero_motore_processato = albero_motore[albero_motore['eliminare']==False]
        albero_motore_processato.drop(columns=['eliminare'], inplace=True)
        albero_motore_processato.reset_index(inplace=True, drop=True)

        # slice PLM_BOM
        slice_1 = PLM_BOM.iloc[0:indice_inizio,]
        slice_3 = PLM_BOM.iloc[indice_fine:,]


        # aggiungere albero motore processato
        # questa plm bom valida solo se c'è motore, altrimenti 
        PLM_BOM = pd.concat([slice_1,albero_motore_processato,slice_3], ignore_index=True) # questo non va bene, perchè accodato in fondo è su X

        


    #**********ATTENZIONE********** SOLO PER CONFRONTO: SUI LIVELLI 2 DEL PLM SENZA GT (PERCHè GRUPPI MANUFACTURING) PRENDO GT DA SAP

    db_gruppi_tecnici_SAP = SAP_BOM[SAP_BOM['Liv.']==2]
    db_gruppi_tecnici_SAP = db_gruppi_tecnici_SAP[['Articolo','Gruppo Tecnico','Descr. Gruppo Tecnico']]
    db_gruppi_tecnici_SAP = db_gruppi_tecnici_SAP.drop_duplicates()

    #prendo i livelli 2 senza gruppo tecnico
    no_gt = list(PLM_BOM[(PLM_BOM['Gruppo Tecnico'].astype(str)=='nan') & (PLM_BOM['Liv.']==2)].Articolo)

    #filtro db_gruppi_tecnici_SAP solo su questi codici, così quando andrò a fare il merge non mi duplica righe
    db_gruppi_tecnici_SAP = db_gruppi_tecnici_SAP[[any(part in codice for part in no_gt) for codice in db_gruppi_tecnici_SAP.Articolo]]
    db_gruppi_tecnici_SAP = db_gruppi_tecnici_SAP.rename(columns={'Gruppo Tecnico':'GT_appoggio','Descr. Gruppo Tecnico':'DGT_appoggio'})

    st.write('Check PLM pre merge descrizione gruppi tecnici livello 2',len(PLM_BOM))
    PLM_BOM['Gruppo Tecnico']=[stringa[:2] for stringa in PLM_BOM['Gruppo Tecnico'].astype(str)] # qui fa diventare na il nan
    PLM_BOM = PLM_BOM.merge(db_gruppi_tecnici_SAP, how='left',left_on='Articolo',right_on='Articolo')

    st.write('Check PLM dopo merge descrizione gruppi tecnici livello 2',len(PLM_BOM))
    PLM_BOM['Gruppo Tecnico'] = np.where((PLM_BOM['Gruppo Tecnico'].astype(str)=='na')&(PLM_BOM['GT_appoggio'].astype(str)=='NO TITOLO'),PLM_BOM.GT_appoggio,PLM_BOM['Gruppo Tecnico'])
    PLM_BOM['Gruppo Tecnico'] = np.where(PLM_BOM['Gruppo Tecnico'].astype(str)=='na',[stringa[:2] for stringa in PLM_BOM.GT_appoggio.astype(str)],PLM_BOM['Gruppo Tecnico'])
    PLM_BOM['Descr. Gruppo Tecnico'] = np.where(PLM_BOM['Descr. Gruppo Tecnico'].astype(str)=='nan',PLM_BOM.DGT_appoggio,PLM_BOM['Descr. Gruppo Tecnico'])

    #correzione gruppi teecnici mancanti
    for i in range(len(PLM_BOM)): 
        if (PLM_BOM['Liv.'].iloc[i] >2) and (PLM_BOM['Gruppo Tecnico'].astype(str).iloc[i]=='na'):
            PLM_BOM['Gruppo Tecnico'].iloc[i] = PLM_BOM['Gruppo Tecnico'].iloc[i-1]
            PLM_BOM['Descr. Gruppo Tecnico'].iloc[i] = PLM_BOM['Descr. Gruppo Tecnico'].iloc[i-1]

    #scrivo errori di eredità gruppo tecnico
    st.subheader('Eccezioni regole Gruppi Tecnici DAA', divider='red')
    st.write('I codici in tabella non hanno potuto ereditare il gruppo tecnico del padre in quanto mancante')
    st.write(PLM_BOM[(PLM_BOM['Gruppo Tecnico'].astype(str)=='na') & (PLM_BOM['Liv.']>2)])


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

#with tab2:

    # Ottengo moto M, V, X 20231228: attenzione, contengono la descrizione
    SAP_M = partizione(lista_SAP_livelli_1[0],SAP_BOM).reset_index(drop=True)
    PLM_M = partizione(lista_PLM_livelli_1[0],PLM_BOM).reset_index(drop=True)
    SAP_V = partizione(lista_SAP_livelli_1[1],SAP_BOM).reset_index(drop=True)
    PLM_V = partizione(lista_PLM_livelli_1[1],PLM_BOM).reset_index(drop=True)
    SAP_X = partizione(lista_SAP_livelli_1[2],SAP_BOM).reset_index(drop=True)
    PLM_X = partizione(lista_PLM_livelli_1[2],PLM_BOM).reset_index(drop=True)

    # 20231228 probabilmente questi df non serviranno
    SAP_BOM_con_descrizione = importa_cap_con_descrizone(SAP_cap)

    SAP_M_descrizione = partizione(lista_SAP_livelli_1[0],SAP_BOM_con_descrizione).reset_index(drop=True) ####
    SAP_V_descrizione = partizione(lista_SAP_livelli_1[1],SAP_BOM_con_descrizione).reset_index(drop=True)
    SAP_X_descrizione = partizione(lista_SAP_livelli_1[2],SAP_BOM_con_descrizione).reset_index(drop=True)

    # Tabelle comparative
    # ok 20231228
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

with tab2: #Action Item livelli 2


    # Scegliere il livello da analizzare (M,V,X) in streamlit----------------------------------------------------------lo metto nella sidebar
        
    st.sidebar.header('Selezionare livello da analizzare', divider = 'red')
    livello_1 = st.sidebar.radio ('Scegli il tipo di distinta livello 1', ['M','V','X'], index=None)
    if not livello_1:
        st.stop()


    # aggiunto [['Liv.','Articolo','Qty']] 20231228
    if livello_1 == 'M':
        L_BOM_input = SAP_M[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
        R_BOM_input = PLM_M[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
    elif livello_1 == 'V':
        L_BOM_input = SAP_V[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
        R_BOM_input = PLM_V[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
    else:
        L_BOM_input = SAP_X[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
        R_BOM_input = PLM_X[['Liv.','Articolo','Qty','Gruppo Tecnico']] 




    st.header(f'Analisi moto {livello_1}', divider = 'red')

    # 20231228 da verificare, forse non è mai utilizzata
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
    # 20231228 ok se alimentata dopo scelta ldel livello modificata
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


    check_ok = len(comparelev_2[comparelev_2['Esito check codice'] == 'Check ok'])
    #st.write('check ok', check_ok)
    non_in_sap = len(comparelev_2[comparelev_2['Esito check codice'] == 'Non in SAP'])
    #st.write('non in sap', non_in_sap)
    non_in_plm = len(comparelev_2[comparelev_2['Esito check codice'] == 'Non in PLM'])
    #st.write('non in plm', non_in_plm)

    kpi_liv2 = (non_in_plm + non_in_sap)/(check_ok+non_in_sap+non_in_plm)*100
    lista_Venn=[check_ok,non_in_plm, non_in_sap]

    # Diagramma di Venn
    fig= plt.figure(figsize=(12,6))
    venn2(subsets =
        (lista_Venn[1],lista_Venn[2],lista_Venn[0]),
        set_labels=('SAP liv.2','PLM liv.2'),
        alpha=0.5,set_colors=('red', 'yellow'))

    df_confronto_lev2 = comparelev_2[comparelev_2['Esito check codice'] != 'Check ok']
    df_confronto_lev2[['Azione','Responsabile','Due date','Status']] = ""

    #st.write(df_confronto_lev2)
    #st.stop()

    # Aggiunto algo per descrizione e gruppo tecnico

    if livello_1 == 'M':
        df_SAP_descrizione = SAP_M_descrizione[['Articolo','Testo breve oggetto','Gruppo Tecnico']]
    elif livello_1 == 'V':
        df_SAP_descrizione = SAP_V_descrizione[['Articolo','Testo breve oggetto','Gruppo Tecnico']]
    else:
        df_SAP_descrizione = SAP_X_descrizione[['Articolo','Testo breve oggetto','Gruppo Tecnico']]


    df_PLM_descrizione = PLM_BOM[['Articolo','Testo breve oggetto']]  #[['Item Id','Rev Name']]----------------------------------------------------------------------Qui scrive la tabella ∆ quantity

    codici_non_in_PLM = df_confronto_lev2[df_confronto_lev2['Esito check codice']=='Non in PLM']
    AIR_SAP = codici_non_in_PLM.merge(df_SAP_descrizione,how='left',left_on='Articolo', right_on='Articolo')
    AIR_SAP = AIR_SAP[['Articolo','Testo breve oggetto','Gruppo Tecnico','Esito check codice','Azione','Responsabile','Due date','Status']]
    AIR_SAP.rename(columns={'Testo breve oggetto': 'Descrizione'}, inplace=True)

    codici_non_in_SAP = df_confronto_lev2[df_confronto_lev2['Esito check codice']=='Non in SAP']
    AIR_PLM = codici_non_in_SAP.merge(df_PLM_descrizione,how = 'left',left_on='Articolo', right_on='Articolo')
    AIR_PLM = AIR_PLM[['Articolo','Testo breve oggetto','Gruppo Tecnico_PLM','Esito check codice','Azione','Responsabile','Due date','Status']]
    AIR_PLM.rename(columns={'Testo breve oggetto':'Descrizione'},inplace=True)



    df_confronto_lev2_descrizione = pd.concat([AIR_SAP,AIR_PLM],ignore_index=True)
    df_confronto_lev2_descrizione.drop_duplicates(inplace=True)
    df_confronto_lev2_descrizione.reset_index(drop=True, inplace=True)

    df_confronto_lev2_descrizione_print = df_confronto_lev2_descrizione[['Articolo','Descrizione','Gruppo Tecnico','Gruppo Tecnico_PLM','Esito check codice']]


    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader('Action Item livelli 2', divider = 'red')
        st.write(df_confronto_lev2_descrizione_print)
        st.markdown('**KPI 1: percentuale livelli 2 ok: :green[{:0.1f}%]**'.format(100-kpi_liv2)) #0,. per separatore miglialia

    with col2:
        st.subheader('Venn livelli 2', divider = 'red')
        st.pyplot(fig)
        st.write('Livelli 2 in SAP e **non in PLM**: ',lista_Venn[1]) ### modificato
        st.write('Livelli 2 in comune: ',lista_Venn[0]) ### modificato
        st.write('Livelli 2 in PLM e **non in SAP**: ',lista_Venn[2]) ### modificato


    # Download file csv

    def convert_df(df):
        return df.to_csv(index=False,decimal=',').encode('utf-8')   # messo index=False
    csv = convert_df(df_confronto_lev2_descrizione)
    st.download_button(
        label="Download Registro delle azioni livelli 2 in CSV",
        data=csv,
        file_name='Registro_Azioni_Livelli_2.csv',
        mime='text/csv',
    )
    #------------------------------------------------------------------------------------- Visualizzazione dei flag | 2024 02 06 AB 

    st.subheader('Confronto configurazione flag',divider='blue')

    delta = df_confronto_lev2_descrizione
    codice = st.selectbox('selezionare codice', delta.Articolo.unique())
    d33 = SAP_cap
    daa = PLM

    colonne_d33 = d33.columns
    print_33 = pd.DataFrame(columns=colonne_d33)
    inizio = 0
    fine = 0
    for i in range(len(d33)):
        codice_check = d33.Materiale.iloc[i]
        if codice_check == codice:
            livello = d33['Liv. esplosione'].iloc[i]
            livello = int(livello[-1:])
            if livello == 2:
                print_33 = pd.concat([print_33,d33[i:i+1]])
                continue

            for j in range(i):
                riga = i-j
                livello_check = int(d33['Liv. esplosione'].iloc[riga][-1:])
                if livello_check == livello - 1:
                    inizio = riga
                    break
            
            for k in range(len(d33)-i):
                riga = i+k
                livello_check = int(d33['Liv. esplosione'].iloc[riga][-1:])
                if livello_check == livello:
                    fine = riga 
                    break

            print_33 = pd.concat([print_33,d33[inizio:fine+1]])

    colonne_daa = daa.columns
    print_aa = pd.DataFrame(columns=colonne_daa)
    inizio_aa = 0
    fine_aa = 0
    for i in range(len(daa)):
        codice_check = daa['Numero componenti'].iloc[i]
        if codice_check == codice:
            livello = daa['Liv. esplosione'].iloc[i]
            livello = int(livello[-1:])
            if livello == 2:
                print_aa = pd.concat([print_aa,daa[i:i+1]])
                continue

            for j in range(i):
                riga = i-j
                livello_check = int(daa['Liv. esplosione'].iloc[riga][-1:])
                if livello_check == livello - 1:
                    inizio = riga
                    break
            
            for k in range(len(daa)-i):
                riga = i+k
                livello_check = int(daa['Liv. esplosione'].iloc[riga][-1:])
                if livello_check == livello:
                    fine = riga 
                    break

            print_aa = pd.concat([print_aa,daa[inizio:fine+1]])
            
    #----------------------stampa

    print_33 = print_33[['Liv. esplosione','Materiale','Testo breve oggetto','Qtà comp. (UMC)','MerceSfusa (BOM)','Ril.Prod.',
                        'Ril.Tecn.','Ril.Ric.','Gruppo Tecnico','Descr. Gruppo Tecnico','Inizio validità','Fine validità']]

    colonne_33 = print_33.columns

    print_aa = print_aa[['Liv. esplosione','Numero componenti','Testo breve oggetto','Qtà comp. (UMC)','Merce sfusa','Rilevante produzione',
                        'Ril. progettazione','Cd.parte di ricambio','Gruppo Tecnico','Descr. Gruppo Tecnico','Inizio validità','Fine validità']] 

    colonne_aa = print_aa.columns

    transcodifica = dict(zip(colonne_aa,colonne_33))
    print_aa = print_aa.rename(columns=transcodifica)

    print_aa['ambiente']='TC'
    print_33['ambiente']='SAP'

    colonne_33 = print_33.columns

    completo = pd.DataFrame(columns=colonne_33)
    completo = pd.concat([print_33,print_aa])
    completo = completo[['ambiente','Liv. esplosione','Materiale','Testo breve oggetto','Qtà comp. (UMC)','MerceSfusa (BOM)','Ril.Prod.',
                        'Ril.Tecn.','Ril.Ric.','Gruppo Tecnico','Descr. Gruppo Tecnico','Inizio validità','Fine validità']]

    def highlight_SAP(s):
        return ['background-color: blue']*len(s) if s.ambiente=='SAP' else ['background-color: black']*len(s)

    #st.subheader('Comparazione',divider='blue')
    st.write('Comparazione flag')
    st.dataframe(completo.style.apply(highlight_SAP, axis=1),width=2500)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------fine confronto flag

with tab3: #Delta qty livelli2

    ##### Confronto qty livelli 2
    # Aggiunto algo per qty

    if livello_1 == 'M':
        df_SAP_des_qty = SAP_M_descrizione[['Liv.','Articolo','Qty','Testo breve oggetto','Gruppo Tecnico']]
    elif livello_1 == 'V':
        df_SAP_des_qty = SAP_V_descrizione[['Liv.','Articolo','Qty','Testo breve oggetto','Gruppo Tecnico']]
    else:
        df_SAP_des_qty = SAP_X_descrizione[['Liv.','Articolo','Qty','Testo breve oggetto','Gruppo Tecnico']]


    df_PLM_2 = R_BOM_input[R_BOM_input['Liv.']==2]

    ##st.write(df_PLM_2)
    #st.stop()
    # uniformo i gruppi tecnici 
    df_PLM_2['Gruppo Tecnico'] = np.where(df_PLM_2['Gruppo Tecnico']!='NO TITOLO', [stringa + '.' for stringa in df_PLM_2['Gruppo Tecnico']], df_PLM_2['Gruppo Tecnico'])
    sum_PLM_2 = df_PLM_2.groupby(by=['Liv.','Articolo','Gruppo Tecnico'],as_index=False).sum()
    sum_PLM_2['key'] = sum_PLM_2['Articolo']+'|'+sum_PLM_2['Gruppo Tecnico']
    sum_PLM_2_tot = df_PLM_2[['Articolo','Qty']].groupby(by='Articolo',as_index=False).sum()
    sum_PLM_2_tot = sum_PLM_2_tot.merge(sum_PLM_2,how='left',left_on='Articolo',right_on='Articolo')


    df_SAP_2=df_SAP_des_qty[df_SAP_des_qty['Liv.']==2]
    sum_SAP_2 = df_SAP_2.groupby(by=['Liv.','Articolo','Testo breve oggetto','Gruppo Tecnico'],as_index=False).sum()
    sum_SAP_2['key'] = sum_SAP_2['Articolo']+'|'+sum_SAP_2['Gruppo Tecnico']
    sum_SAP_2_tot = df_SAP_2[['Articolo','Qty']].groupby(by='Articolo',as_index=False).sum()
    sum_SAP_2_tot = sum_SAP_2_tot.merge(sum_SAP_2,how='left',left_on='Articolo',right_on='Articolo')


    out = sum_SAP_2_tot.merge(sum_PLM_2_tot,how='outer',left_on='key',right_on='key')

    out = out.rename(columns={'Articolo_x':'Articolo','Qty_x_x':'Qty_tot_SAP','Liv._x':'Liv_SAP','Gruppo Tecnico_x':'Gruppo Tecnico_SAP','Qty_y_x':'Qty Gruppo Tecnico_SAP',
                            'Qty_x_y':'Qty_tot_PLM','Liv._y':'Liv_PLM','Gruppo Tecnico_y':'Gruppo Tecnico_PLM','Qty_y_y':'Qty Gruppo Tecnico_PLM'})

    out['Delta_totale']=out['Qty_tot_SAP']-out['Qty_tot_PLM']
    out['Delta_Gruppo_Tecnico']=out['Qty Gruppo Tecnico_SAP']-out['Qty Gruppo Tecnico_PLM']
    out = out[(out['Delta_totale']!=0) | (out['Delta_Gruppo_Tecnico']!=0)]

    #tolgo i not in PLPM o SAP
    out = out[out.Articolo.astype(str)!='nan']
    codici_outer = list(df_confronto_lev2_descrizione_print.Articolo)
    out = out[[all(codice not in articolo for codice in codici_outer) for articolo in out.Articolo.astype(str) ]]
    out = out.drop(columns=['key','Articolo_y'])


    # Confronto quantità totali
    st.subheader('Delta quantità codici livello 2',divider='red')
    st.dataframe(out,width=2500)

with tab4:

#confronto campi custom 20240129#–---------------------------------------------------------------------------------------------------------------------------------

    if livello_1 == 'M':
        L_BOM_input_custom = SAP_M[['Liv.','Articolo','Qty','Gruppo Tecnico','Testo posizione riga 1','Testo posizione riga 2','STGR','Descrizione Sottogruppo']] #---------------------per confronto custom
        R_BOM_input_custom = PLM_M[['Liv.','Articolo','Qty','Gruppo Tecnico','Testo posizione riga 1','Testo posizione riga 2','STGR','Descrizione Sottogruppo']] 
    elif livello_1 == 'V':
        L_BOM_input_custom = SAP_V[['Liv.','Articolo','Qty','Gruppo Tecnico','Testo posizione riga 1','Testo posizione riga 2','STGR','Descrizione Sottogruppo']] #---------------------per confronto custom
        R_BOM_input_custom = PLM_V[['Liv.','Articolo','Qty','Gruppo Tecnico','Testo posizione riga 1','Testo posizione riga 2','STGR','Descrizione Sottogruppo']]
    else:
        L_BOM_input_custom = SAP_X[['Liv.','Articolo','Qty','Gruppo Tecnico','Testo posizione riga 1','Testo posizione riga 2','STGR','Descrizione Sottogruppo']] #---------------------per confronto custom
        R_BOM_input_custom = PLM_X[['Liv.','Articolo','Qty','Gruppo Tecnico','Testo posizione riga 1','Testo posizione riga 2','STGR','Descrizione Sottogruppo']]


    df_descrizioni = SAP_raw[['Articolo','Testo breve oggetto']]
    df_descrizioni = df_descrizioni.drop_duplicates()

    #st.write('Prima del merge', len(L_BOM_input_custom))
    L_BOM_input_custom = L_BOM_input_custom.merge(df_descrizioni,how='left',left_on='Articolo',right_on='Articolo')
    #st.write('Dopo il merge', len(L_BOM_input_custom))

    df_SAP_2_custom = L_BOM_input_custom[L_BOM_input_custom['Liv.']==2]
    df_SAP_2_custom['key']=df_SAP_2_custom['Articolo']+' | '+df_SAP_2_custom['Gruppo Tecnico']

    df_PLM_2_custom = R_BOM_input_custom[R_BOM_input_custom['Liv.']==2]
    df_PLM_2_custom['Gruppo Tecnico'] = np.where(df_PLM_2_custom['Gruppo Tecnico']!='NO TITOLO', [stringa + '.' for stringa in df_PLM_2_custom['Gruppo Tecnico']], df_PLM_2_custom['Gruppo Tecnico'])
    df_PLM_2_custom['key']=df_PLM_2_custom['Articolo']+' | '+df_PLM_2_custom['Gruppo Tecnico']

    custom_compare = df_SAP_2_custom.merge(df_PLM_2_custom,how='left',left_on='key',right_on='key',suffixes=('_SAP','_PLM'))
    custom_compare = custom_compare.drop(columns=['key','Liv._SAP','Qty_SAP','Liv._PLM','Qty_PLM'])
    custom_compare = custom_compare[['Articolo_SAP','Testo breve oggetto','Gruppo Tecnico_SAP','Gruppo Tecnico_PLM','Testo posizione riga 1_SAP','Testo posizione riga 1_PLM',
                                    'Testo posizione riga 2_SAP','Testo posizione riga 2_PLM','STGR_SAP','STGR_PLM','Descrizione Sottogruppo_SAP','Descrizione Sottogruppo_PLM']]

    custom_compare['Testo posizione riga 1_PLM']=[str.lower(testo) for testo in custom_compare['Testo posizione riga 1_PLM'].astype(str)]
    custom_compare['check_testo1']=custom_compare['Testo posizione riga 1_PLM']==custom_compare['Testo posizione riga 1_SAP']
    custom_compare['check_testo2']=custom_compare['Testo posizione riga 2_PLM']==custom_compare['Testo posizione riga 2_SAP']
    custom_compare['check_STGR']=custom_compare['STGR_PLM'].astype(str)==custom_compare['STGR_SAP'].astype(str)
    custom_compare['check_STGR_desc']=custom_compare['Descrizione Sottogruppo_PLM'].astype(str)==custom_compare['Descrizione Sottogruppo_SAP'].astype(str)


    # rimozione dei duplicati
    # creazione chiave codice + | + gruppo Tecnico + | + testo riga 1
    # stessa cosa la faccio anche sul SAP (già sul frame df_SAP_2_custom)
    # nuova colonna in PLM: più somigliante di SAP
    # merge con sap sulla chiave per ottenere STGR

    #custom_compare=custom_compare[(custom_compare.check_testo1==False) | (custom_compare.check_testo2==False) | (custom_compare.check_STGR==False)  | (custom_compare.check_STGR_desc==False) ]
    prima = len(custom_compare)
    custom_compare_testo1=custom_compare[(custom_compare.check_testo1==False)]
    custom_compare_STGR=custom_compare[(custom_compare.check_STGR==False) | (custom_compare.check_STGR_desc==False)]
    dopo_testo1 = len(custom_compare_testo1)
    dopo_STGR = len(custom_compare_STGR)

    st.subheader('Delta campi custom', divider='red')

    #st.write('Righe non omogenee per testo riga 1',custom_compare_testo1)
    #st.write('Righe totali: :grey[{}] | Righe con campi custom omogenei per testo riga 1: :green[ {} ] | Righe da correggere: :red[{} ]'.format(prima, prima-dopo_testo1, dopo_testo1))
    #st.divider()
    st.write('Righe non omogenee per STGR / Descrizione STGR',custom_compare_STGR)
    st.write('Righe totali: :grey[ {}] | Righe con campi custom omogenei per Sottogruppi: :green[{}]| Righe da correggere: :red[{}] '.format(prima, prima-dopo_STGR, dopo_STGR))

    #st.write('Righe totali: {} | Righe con campi custom omogenei: {} | Righe da correggere:{} '.format(prima, prima-dopo, dopo))
    st.divider()

    # implementazione logica Fuzzy

  #  fuzzy_PLM = custom_compare[['Articolo_SAP','Testo breve oggetto','Gruppo Tecnico_PLM','Testo posizione riga 1_PLM',
  #                              'STGR_PLM','Descrizione Sottogruppo_PLM']]
   # fuzzy_PLM = fuzzy_PLM.drop_duplicates()


  #  fuzzy_PLM['key_fuzzy']=np.where(fuzzy_PLM['Testo posizione riga 1_PLM'] != 'nan',
  #                          fuzzy_PLM['Articolo_SAP']+'|'+fuzzy_PLM['Gruppo Tecnico_PLM']+'|'+fuzzy_PLM['Testo posizione riga 1_PLM'],
   #                         fuzzy_PLM['Articolo_SAP']+'|'+fuzzy_PLM['Gruppo Tecnico_PLM'])
  #  fuzzy_PLM['best_fit'] = None


 #   fuzzy_SAP = df_SAP_2_custom.copy()
 #   fuzzy_SAP['key_fuzzy'] = np.where(fuzzy_SAP['Testo posizione riga 1'].astype(str) != 'nan',
  #                          fuzzy_SAP['Articolo']+'|'+fuzzy_SAP['Gruppo Tecnico']+'|'+fuzzy_SAP['Testo posizione riga 1'],
  #                          fuzzy_SAP['Articolo']+'|'+fuzzy_SAP['Gruppo Tecnico'])




  #  fuzzy_SAP_un=fuzzy_SAP['key_fuzzy']#.drop_duplicates(inplace=True)
  #  fuzzy_SAP_un = list(fuzzy_SAP_un.drop_duplicates())




   # st.write('SAP no dupl',fuzzy_SAP_un)



    #for i in range(len(fuzzy_PLM)):

     #   key_check = dict(zip(fuzzy_SAP_un,[0 for i in range(len(fuzzy_SAP_un))])) #azzero il dizionario dei punteggi
     #   chiave = fuzzy_PLM.key_fuzzy.iloc[i]
     #   for k in range(len(key_check)):
      #      chiave_dic = fuzzy_SAP_un[k]
        # try:
     #       key_check[chiave_dic] = fuzz.ratio(str(chiave).lower(), chiave_dic)
            #except:
            #  st.write('problema')
            #   key_check[chiave_dic] = 0

      #  best_score = max(key_check.values())
     #   sub_dic={k: v for k,v in key_check.items() if v == best_score}
       # best_score_item = list(sub_dic.keys())[0]
       # fuzzy_PLM['best_fit'].iloc[i]=str.lower(best_score_item)
        #key_check = {}
        
            
   # st.write('PLM',fuzzy_PLM)
   # fuzzy_SAP['key_fuzzy'] = [str.lower(key) for key in fuzzy_SAP['key_fuzzy'].astype(str)]
  #  fuzzy_SAP = fuzzy_SAP[fuzzy_SAP.STGR.astype(str) != 'nan' ]
  #  st.write('fuzzy SAP',fuzzy_SAP)
   # fuzzy_PLM = fuzzy_PLM.merge(fuzzy_SAP[['key_fuzzy','STGR']], how='left', left_on='best_fit',right_on='key_fuzzy')

   # fuzzy_PLM = fuzzy_PLM[(fuzzy_PLM.STGR_PLM.astype(str)=='nan') & (fuzzy_PLM.STGR.astype(str)!='nan') ]
  #  st.write('Output')
  #  st.dataframe(fuzzy_PLM, width=2500)


    #st.stop()


#st.stop()#------------------------------------------------------------------------------------------------------------------------------------------------ STOP 1 ---- LAVORO FINO AD ANALISI ∆STGR



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

df_livelli_2_ko_per_grafo['Isomorfi']="D"
df_livelli_2_ko_per_grafo.reset_index(inplace = True, drop=True)

def test_albero (tree):   # aggiunta funzione per test integrità albero
    albero_integro = True
    for i in range (len (tree)):
        if i+1 == len(tree):
            break
        else:
            livello = tree.loc[i,'Liv.']
            livello_successivo = tree.loc[i+1,'Liv.']
            if (livello_successivo-livello) > 1:
                albero_integro = False
                break
    return albero_integro

for i in range(len(df_livelli_2_ko_per_grafo)):
    art = df_livelli_2_ko_per_grafo.loc[i,'Articolo']    
    # inizio modifica per verifica integrità albero
    albero_L = test_albero(albero(art,L_BOM_input))
    albero_R = test_albero(albero(art,R_BOM_input))
    if not (albero_L and albero_R):
        continue
    # fine modifica
    df_livelli_2_ko_per_grafo.loc[i,'Isomorfi']=isomorfi_3(art,L_BOM_input,R_BOM_input)  

per_grafo = df_livelli_2_ko_per_grafo[df_livelli_2_ko_per_grafo['Isomorfi']==False]
alberi_incompleti = df_livelli_2_ko_per_grafo[df_livelli_2_ko_per_grafo['Isomorfi']=='D']
alberi_incompleti=alberi_incompleti['Articolo']
codici_grafo = per_grafo['Articolo'] # serie per selezione codici
numero_codici_grafo = len(codici_grafo)
numero_codici_isomorfi=len(df_livelli_2_ko_per_grafo[df_livelli_2_ko_per_grafo['Isomorfi']==True])
numero_alberi_incompleti = len(alberi_incompleti)

st.header(f'Analisi moto {livello_1} livelli 2 con figli', divider = 'red' )
st.write(f'Codici con alberi isomorfi: {numero_codici_isomorfi} | **non necessitano di analisi**')
st.divider()
st.write(f'Codici con alberi incompleti: {numero_alberi_incompleti}')
st.markdown(':red[Controllare attributi Merce sfusa, Rilevanza Produttiva, Rilevanza Tecnica, Rilevanza Ricambi]')
st.dataframe(alberi_incompleti)

#------------------------------------------------------------------------------------------------------------------------------------SEGNALIBRO_ inizio modifica AB | 19-11-2024 | articoli segati 

def highlight_rimossi(s):
    return ['background-color: red']*len(s) if s.rimosso=='si' else ['background-color: black']*len(s)

def highlight_rimossi_daa(s):
    return ['background-color: blue']*len(s) if s.rimosso=='si' else ['background-color: black']*len(s)

codice_albero_incompleto = st.selectbox('selezionare codice', alberi_incompleti)
if not codice_albero_incompleto:
    st.stop()

albero_33 = partizione(codice_albero_incompleto,SAP_raw) # estrazione albero da distinta grezza (con tutti i componenti)
albero_33_incompleto = partizione(codice_albero_incompleto,SAP_BOM) # estrazione albero da distinta elaborata (componenti segati)

#Confronto distinte

articoli_raw = albero_33.Articolo #lista articoli db completa
articoli_incompleto = albero_33_incompleto.Articolo #lista articoli db segata

diff = set(articoli_raw).difference(articoli_incompleto) # intersezione delle due liste
albero_33['rimosso']=None
for articolo in diff:
    for i in range(len(albero_33)):
        if  albero_33.Articolo.iloc[i] == articolo:
            albero_33.rimosso.iloc[i]='si'


albero_daa = partizione(codice_albero_incompleto,PLM_raw) # estrazione albero da distinta grezza (con tutti i componenti)
albero_daa_incompleto = partizione(codice_albero_incompleto,PLM_BOM) # estrazione albero da distinta elaborata (componenti segati)

articoli_raw_daa = albero_daa.Articolo #lista articoli db completa
articoli_incompleto_daa = albero_daa_incompleto.Articolo #lista articoli db segata

diff_daa = set(articoli_raw_daa).difference(articoli_incompleto_daa)
albero_daa['rimosso']=None
for articolo in diff_daa:
    for i in range(len(albero_daa)):
        if  albero_daa.Articolo.iloc[i] == articolo:
            albero_daa.rimosso.iloc[i]='si'

st.subheader('Analisi alberi incompleti', divider = 'grey')
st.write('Le righe evidenziate sono state rimosse durante il flusso di analisi')
st.subheader('D33', divider='red') 
st.dataframe(albero_33.style.apply(highlight_rimossi, axis=1), width=2500)
st.subheader('DAA', divider = 'blue')
st.dataframe(albero_daa.style.apply(highlight_rimossi_daa, axis=1), width=2500)

# costruzione tabella di output
colonne_compare=albero_daa.columns
comparazione_alberi = pd.DataFrame(columns=colonne_compare)
comparazione_alberi['Ambiente']=None
comparazione_alberi['Padre livello 2']=None

for i in range(len(albero_33)):
    codice_33 = albero_33['Articolo'].iloc[i]
    for j in range(len(albero_daa)):
        codice_daa = albero_daa['Articolo'].iloc[j]
        if codice_daa == codice_33:
            if albero_33.rimosso.iloc[i] != albero_daa.rimosso.iloc[j]:
                l = len(comparazione_alberi)
                comparazione_alberi.loc[l]=albero_33.iloc[i]
                comparazione_alberi['Padre livello 2'].loc[l]= codice_albero_incompleto
                comparazione_alberi['Ambiente'].loc[l]='SAP'
                l = len(comparazione_alberi)
                comparazione_alberi.loc[l]=albero_daa.iloc[j]
                comparazione_alberi['Padre livello 2'].loc[l]= codice_albero_incompleto
                comparazione_alberi['Ambiente'].loc[l]='PLM'
        
elenco_codici = set(list(albero_33.Articolo))

for i in range(len(albero_daa)):
    codice_daa = albero_daa['Articolo'].iloc[i]
    if codice_daa not in elenco_codici:
        st.write('warning',codice_daa)
        for j in range(len(albero_daa)):
            codice_d33 = albero_33['Articolo'].iloc[j]
            if codice_33 == codice_daa:
                if albero_daa.rimosso.iloc[i] != albero_33.rimosso.iloc[j]:
                    l = len(comparazione_alberi)
                    comparazione_alberi.loc[l]=albero_33.iloc[j]
                    comparazione_alberi['Padre livello 2'].loc[l]= codice_albero_incompleto
                    comparazione_alberi['Ambiente'].loc[l]='SAP'
                    l = len(comparazione_alberi)
                    comparazione_alberi.loc[l]=albero_daa.iloc[i]
                    comparazione_alberi['Padre livello 2'].loc[l]= codice_albero_incompleto
                    comparazione_alberi['Ambiente'].loc[l]='PLM'

comparazione_alberi = comparazione_alberi[['Padre livello 2','Ambiente','Liv.','Articolo','Testo breve oggetto','Qty','MerceSfusa (BOM)','Ril.Tecn.','Ril.Prod.',
                                           'Ril.Ric.','rimosso','Gruppo Tecnico','Descr. Gruppo Tecnico','Testo posizione riga 1','Testo posizione riga 2','STGR',
                                           'Descrizione Sottogruppo','Gruppo appartenenza','Descr. Gruppo Appartenenza']]
st.write(comparazione_alberi)

moto = SAP_cap.Materiale.iloc[0]

#if st.button('download_excel'):
#    comparazione_alberi.to_excel('/Users/Alessandro/Documents/AB/Clienti/ADI!/Ducati/DeltaBOM/Download/{} - livello{} - {}.xlsx'.format(moto,livello_1, codice_albero_incompleto ))


# Serve vedere le distinte di uscita appaiate TO_DO_1 26
    
#------------------------------------------------------------------------------------------------------------------------------------SEGNALIBRO_ fine modifica AB | 19-11-2024 | articoli segati 
    
st.divider()
st.write(f'Codici con alberi non isomorfi: {numero_codici_grafo}')
st.divider()

kpi_liv_2_con_figli = ((numero_codici_grafo+numero_alberi_incompleti)/(numero_codici_grafo+numero_codici_isomorfi+numero_alberi_incompleti)*100)
st.markdown('**KPI 2: percentuale ok livelli 2 con figli: :green[{:0.1f}%]**'.format(100-kpi_liv_2_con_figli)) #0,. per separatore miglialia

st.subheader('Tabella delta BOM per codici con alberi non isomorfi')

df_delta_complessivo = pd.DataFrame()
for codice in codici_grafo:
    albero_L = albero(codice,L_BOM_input) # SAP_M (ad esempio)
    albero_R = albero(codice,R_BOM_input) # PLM_M (ad esempio)
    df_output_confronto_L = albero_L
    df_output_confronto_L['Liv.']=df_output_confronto_L['Liv.']+2
    df_output_confronto_L.rename(columns= {'Liv.':'Liv. SAP','Articolo':'Articolo SAP','Qty':'Qty SAP'},inplace=True)
    # aggiunge descrizione e gruppo tecnico
 
    #nel merge escludo la colonna Gruppo tecnico perchè è in etrambi, in questo modo non me le chiama x e y e ne tiene una soltanto
    AIR_SAP_sub = df_output_confronto_L.merge(SAP_M.loc[:,SAP_M.columns != 'Gruppo Tecnico'],how='left',left_on='Articolo SAP', right_on='Articolo') 

    AIR_SAP_sub = AIR_SAP_sub[['Liv. SAP','Articolo SAP','Testo breve oggetto','Gruppo Tecnico','Qty SAP']]
    AIR_SAP_sub.rename(columns={'Testo breve oggetto':'Descrizione SAP'},inplace=True)
    AIR_SAP_sub.drop_duplicates(inplace=True)
    AIR_SAP_sub.reset_index(drop=True, inplace=True)

    df_output_confronto_R = albero_R
    df_output_confronto_R['Liv.']=df_output_confronto_R['Liv.']+2
    df_output_confronto_R.rename(columns= {'Liv.':'Liv. PLM','Articolo':'Articolo PLM','Qty':'Qty PLM'},inplace = True)
    # aggiunge descrizione
    #AIR_PLM_sub = df_output_confronto_R.merge(df_PLM_descrizione,how = 'left',left_on='Articolo PLM', right_on='Item Id')
    AIR_PLM_sub = df_output_confronto_R.merge(PLM_M,how = 'left',left_on='Articolo PLM', right_on='Articolo')
    AIR_PLM_sub = AIR_PLM_sub.drop(columns='Articolo')
    AIR_PLM_sub = AIR_PLM_sub[['Liv. PLM','Articolo PLM','Testo breve oggetto','Qty PLM']]
    AIR_PLM_sub.rename(columns={'Testo breve oggetto':'Descrizione PLM'},inplace=True)
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

st.subheader('Grafo per codici di livello 2 con alberi non isomorfi')
part_number = st.selectbox('Seleziona articolo da analizzare',codici_grafo,index=None )
if not part_number:
    st.stop()

# Sceglie BOM in base al livello moto

if livello_1 == 'M':
    L_BOM_input_compare = SAP_M[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
    R_BOM_input_compare = PLM_M[['Liv.','Articolo','Qty', 'Gruppo Tecnico']] 
elif livello_1 == 'V':
    L_BOM_input_compare = SAP_V[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
    R_BOM_input_compare = PLM_V[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
else:
    L_BOM_input_compare = SAP_X[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
    R_BOM_input_compare = PLM_X[['Liv.','Articolo','Qty','Gruppo Tecnico']]  

albero_L = albero(part_number,L_BOM_input_compare) # SAP_M (ad esempio)#------------------------------------------------------------------------------------------------------------
albero_R = albero(part_number,R_BOM_input_compare) # PLM_M (ad esempio)

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


df_output_confronto_R = albero_R
df_output_confronto_R['Liv.']=df_output_confronto_R['Liv.']+2
df_output_confronto_R.rename(columns= {'Liv.':'Liv. PLM','Articolo':'Articolo PLM','Qty':'Qty PLM'},inplace = True)

st.subheader('Grafo SAP | PLM', divider = 'red')
#st.write('grafo disablitato momentaneamente')
fig_a = disegna_grafi(part_number,L_BOM_input,R_BOM_input)
st.pyplot(fig_a)


albero_L_compare = albero_L.merge(anagrafica_sap, how='left',left_on='Articolo SAP',right_on='Materiale')
albero_R_compare = albero_R.merge(anagrafica_plm, how='left',left_on='Articolo PLM',right_on='Numero componenti')

def compara_strutture(albero_SAP, albero_PLM):
    
    albero_SAP = albero_SAP.rename(columns={'Liv. SAP':'livello','Articolo SAP':'articolo','Testo breve oggetto':'descrizione'})
    albero_SAP = albero_SAP[['livello','articolo','descrizione']]
    albero_PLM = albero_PLM.rename(columns={'Liv. PLM':'livello','Articolo PLM':'articolo','Testo breve oggetto':'descrizione'})
    albero_PLM = albero_PLM[['livello','articolo','descrizione']]    
    
    albero_PLM['Liv'] = [int(stringa[-1]) for stringa in albero_PLM['livello'].astype(str)]
    albero_PLM=albero_PLM.drop('livello',axis=1)
    albero_SAP['Liv'] = [int(stringa[-1]) for stringa in albero_SAP['livello'].astype(str)]
    albero_SAP = albero_SAP.drop('livello',axis=1)
    sap_new = albero_SAP.merge(albero_PLM, how='left',left_on='articolo',right_on='articolo')
    sap_new['riportato'] = np.where(sap_new.descrizione_y.astype(str)=='nan','si',None)
    albero_PLM['new_index']=None
    for i in range(len(albero_PLM)):
        albero_PLM['new_index'].iloc[i]=100*i    
        
    sap_new['padre_plm']=None
    for i in range(len(sap_new)):
        n=0
        if sap_new.riportato.iloc[i]=='si':
            for j in range(20):
                if sap_new.riportato.astype(str).iloc[i-j] == 'None':               
                    print(sap_new.riportato.astype(str).iloc[i-j])
                    codice = sap_new.articolo.iloc[i-j]               
                    indice_plm = albero_PLM[albero_PLM.articolo == codice].index[0]
                    sap_new.padre_plm.iloc[i] = indice_plm*100 +1
                    break
    to_append = sap_new[['articolo','descrizione_y','Liv_x','padre_plm']][sap_new.riportato=='si']
    to_append = to_append.rename(columns={'descrizione_y':'descrizione','Liv_x':'Liv','padre_plm':'new_index'})
    albero_PLM = pd.concat([albero_PLM,to_append])  
    albero_PLM = albero_PLM.sort_values(by='new_index')
    plm_new = albero_PLM.merge(albero_SAP, how='left',left_on='articolo',right_on='articolo')
    plm_new = plm_new.rename(columns={'descrizione_x':'descrizione_PLM','Liv_x':'Liv_PLM','descrizione_y':'descrizione_SAP','Liv_y':'Liv_SAP'})
    plm_new = albero_PLM.merge(albero_SAP, how='left',left_on='articolo',right_on='articolo')
    return plm_new

confronto_alberi = compara_strutture(albero_L_compare, albero_R_compare)

st.write(confronto_alberi)

df_delta = df_output_confronto_L.merge(df_output_confronto_R,how='outer', left_on='Articolo SAP', right_on='Articolo PLM',
                             indicator=True,
                                  left_index=False,right_index=False)#,)

df_delta.rename(columns={'_merge':'Delta'}, inplace=True)
df_delta.replace({'left_only':'Solo in SAP','right_only':'Solo in PLM','both':'Comune'},inplace=True)
df_delta['Delta qty SAP-PLM']=df_delta['Qty SAP']-df_delta['Qty PLM']

st.subheader(f'Tabella delta {part_number}', divider='red')
st.write(df_delta)

csv_2 = convert_df(df_delta)
st.download_button(
    label=f"Download confronto {part_number} in CSV",
    data=csv_2,
    file_name=f'{part_number}.csv',
    mime='text/csv',
)