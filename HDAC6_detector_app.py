######################
# Import libraries
######################
import matplotlib.pyplot as plt
from matplotlib import cm
from rdkit.Chem.Draw import SimilarityMaps
from numpy import loadtxt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import joblib
from IPython.display import HTML
from stmol import* 
import py3Dmol
from molvs import standardize_smiles
from math import pi
import ipyspeck
import ipywidgets 

######################
# Page Title
######################
st.write("<h1 style='text-align: center; color: #FF7F50;'> HDAC6 DETECTOR</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align: center; color: #483D8B;'> The application provides an alternative method for assessing the potential of chemicals to be Histone deacetylase 6 (HDAC6) inhibitors.</h3>", unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
col1, col2, col3, col4 = st.columns(4)


with col1:
   st.header("Machine learning")
   st.image("figures/artificial intelligence.png", width=125)
   st.text_area('Text to analyze', '''This application makes predictions based on Quantitative Structure-Activity Relationship (QSAR) models build on curated datasets generated from scientific articles. The  models were developed using open-source chemical descriptors based on Morgan and topological fingerprints, along with the gradient boosting method (GBM) and  multilayer perceptron (MLP) classifier, using Python 3.7''', height=350, label_visibility="hidden" )


with col2:
   st.header("OECD principles")
   st.image("figures/checklist.png", width=125)
   st.text_area('Text to analyze', '''We follow the best practices for model development and validation recommended by guidelines of the Organization for Economic Cooperation and Development (OECD). For assessment of the applicability domain, we used Tanimoto similarity based on Morgan and topological  fingerprints between the test set compound and nearest neighbor in the training set. The contribution maps are generated from QSAR models to predict atoms and fragments that contribute to HDAC6 activity. This method provides a straightforward mechanistic interpretation of the predicted outcomes, assisting users to propose structural modifications to increase the HDAC6 activity of inhibitors''', height=350, label_visibility="hidden" )
# st.write('Sentiment:', run_sentiment_analysis(txt))

with col3:
   st.header("Predictive power")
   st.image("figures/strong.png", width=125)
   st.text_area('Text to analyze', '''The QSAR models showed high predictive power with vigorous validation metrics for test sets, achieving accuracy, sensitivity, and specificity ranging from 88 to 100%. The predictive ability of the developed QSAR models is further confirmed by our synthesis of new HDAC6 inhibitors and in vivo studies. The application HDAC6 DETECTOR allowed to correctly predict the activity class for all 12 new substances''', height=350, label_visibility="hidden" )
# st.write('Sentiment:', run_sentiment_analysis(txt))

with col4:
   st.header("Lipinski’s rule")
   st.image("figures/puzzle-piece.png", width=125)
   st.text_area('**Bioavailability Radar**', '''Estimating the bioavailability of a compound is an important factor in drug development. Lipinski’s rule of five was introduced to estimate the oral bioavailability of a compound. Our bioavailability radar is displayed for a quick assessment of the compliance of the tested compound with the Lipinski rules.''', height=350, label_visibility="hidden" )

with open("manual.pdf", "rb") as file:
    btn=st.download_button(
    label="Click to download brief manual",
    data=file,
    file_name="manual of HDAC6 DETECTOR.pdf",
    mime="application/octet-stream"
)
# Download experimental data
df = pd.read_csv('datasets/HDAC6_exp_data_inchi.csv')
res = (df.groupby("inchi").apply(lambda x: x.drop(columns="inchi").to_dict("records")).to_dict())    
######################
# Main functions
######################
def rdkit_numpy_convert(f_vs):
                output = []
                for f in f_vs:
                    arr = np.zeros((1,))
                    DataStructs.ConvertToNumpyArray(f, arr)
                    output.append(arr)
                    return np.asarray(output)

def getProba(fp, predictionFunction):
    return predictionFunction((fp,))[0][1]


def makeblock(smi):
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                mblock = Chem.MolToMolBlock(mol)
                return mblock

def render_mol(xyz):
                xyzview = py3Dmol.view()#(width=400,height=400)
                xyzview.addModel(xyz,'mol')
                xyzview.setStyle({'stick':{}})
                xyzview.setBackgroundColor('black')
                xyzview.zoomTo()
                showmol(xyzview,height=500,width=500)
def lipinski(smiles):
    mol=Chem.MolFromSmiles(smiles)
    desc_MolWt = Descriptors.MolWt(mol)
    desc_MolLogP = Descriptors.MolLogP(mol)
    desc_NumHDonors = Descriptors.NumHDonors(mol)
    desc_NumHAcceptors = Descriptors.NumHAcceptors(mol)

    df = pd.DataFrame({
    'group': ['A','B'],
    'HBAs/2': [5, desc_NumHAcceptors/2],
    'HBD': [5, desc_NumHDonors],
    'MW/100': [5, desc_MolWt/100],
    'LogP': [5, desc_MolLogP]})
    categories=list(df)[1:]
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
                
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)

    ax.set_rlabel_position(0)
    plt.yticks([1,2,3,4,5,6,7,8,9,10], ["1","2","3",'4','5','6','7','8','9','10'], color="grey", size=7)
    plt.ylim(0, 10)
                
    values=df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="The area of Lipinsik’s rule")
    ax.fill(angles, values, 'b', alpha=0.1)

    values=df.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Values for test substance")
    ax.fill(angles, values, 'r', alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    descriptors = pd.DataFrame({'Molecular weight(MW), Da': [desc_MolWt,500],
                 'Octanol-water coefficient(LogP)': [desc_MolLogP,5], 
                 'Number of hydrogen bond donors (HBD)': [desc_NumHDonors,5],
                  'Number of hydrogen bond acceptors(HBAs)':[desc_NumHAcceptors,10],
                   'Val.': ['Values for the test substance',
                   'Reference value of Lipinsik’s rule']}, index=None).set_index('Val.').T
    
    return st.pyplot(plt),st.dataframe(descriptors)




# Select and read  saved model
models_option = st.sidebar.selectbox('Select QSAR models for prediction', ('GBM_Morgan fingerprints', 'MLP_Topological fingerprints'))

    
if models_option == 'GBM_Morgan fingerprints':
    load_model_GBM = pickle.load(open('Morgan_fingerprint/HDAC6_GBM.pkl', 'rb'))
    st.sidebar.header('Select input molecular files')
    # functions and constants
    threshold = 0.45
    def fpFunction(m, atomId=-1):
        fp = SimilarityMaps.GetMorganFingerprint(m, atomId=atomId, radius=2, nBits=1024)
        return fp
     
    # Read SMILES input
    SMILES = st.sidebar.checkbox('SMILES notations (*.smi)')
    if SMILES:
        SMILES_input = ""
        compound_smiles = st.sidebar.text_area("Enter SMILES", SMILES_input)
        if len(compound_smiles)!=0:
            smiles=standardize_smiles(compound_smiles)
            m = Chem.MolFromSmiles(smiles)
            inchi = str(Chem.MolToInchi(m))
            im = Draw.MolToImage(m)
            st.sidebar.image(im)
        
        
        if st.sidebar.button('PREDICT COMPOUND FROM SMILES'):
            # Calculate molecular descriptors
            f_vs = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, useFeatures=False, useChirality=False)]
            X = rdkit_numpy_convert(f_vs)

            ######################
            # Pre-built model
            ######################

            # Apply model to make predictions
            prediction_GBM = load_model_GBM.predict(X)
            prediction_GBM = np.array(prediction_GBM)
            prediction_GBM = np.where(prediction_GBM == 1, "Active", "Inactive")           


            # Estimination AD
            
            mol = Chem.MolFromSmiles(smiles)
            mg = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)

            d = {}
            for m in Chem.SDMolSupplier('datasets/HDAC6_ws.sdf'):
                if m is not None:
                    mg_ = AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True)
                    d.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(mg, mg_))
            df_ECFP4 = pd.DataFrame.from_dict(d).T
            if df_ECFP4[0].max()>=threshold:
                cpd_AD_vs = "Inside AD"
            else:
                cpd_AD_vs = "Outside AD"

            # search experimental value
            if inchi in res:
                exp=round(res[inchi][0]['pchembl_value_mean'],2)           
                std=round(res[inchi][0]['pchembl_value_std'],4)
                chembl_id=str(res[inchi][0]['molecule_chembl_id'])
                y_pred_con='see experimental value'
                cpd_AD_vs='-'
                
            else:
                y_pred_con=prediction_GBM[0]
                cpd_AD_vs=cpd_AD_vs
                exp="-"
                std="-"
                chembl_id="not detected"
                # Generate maps of fragment contribution
            
                fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(mol, fpFunction, lambda x: getProba(x, load_model_GBM.predict_proba), colorMap=cm.PiYG_r)
                st.write('**Predicted fragments contribution:**')
                st.pyplot(fig)
                st.write('The chemical fragments are colored in green (predicted to reduce inhibitory activity) or magenta (predicted to increase activity HDAC6 inhibitors). The gray isolines separate positive and negative contributions.')

            st.header('**Prediction results:**')


            common_inf = pd.DataFrame({'SMILES':smiles, 'Predicted value, pIC50': y_pred_con, 'Applicability domain': cpd_AD_vs,'Experimental value, pIC50': exp,'Standard deviation': std,
            'Chemble ID': chembl_id}, index=[1])
            predictions_pred=common_inf.astype(str) 
            st.dataframe(predictions_pred)

            
           
            # Lipinski's rule
            st.header("**The Bioavailability Radar: сompliance the Lipinski's rule of five**")
 
            lipinski(compound_smiles)    
          
            # 3D structure
            st.header('**3D structure of the studied compound:**')

            blk=makeblock(compound_smiles)
            render_mol(blk)
            st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')

   

            
    # Read SDF file 
    LOAD = st.sidebar.checkbox('MDL multiple SD file (*.sdf)')
    if LOAD:
        uploaded_file = st.sidebar.file_uploader("Choose a file")
        if uploaded_file is not None:
            st.header('**1. CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:**')
            supplier = Chem.ForwardSDMolSupplier(uploaded_file,sanitize=False)
            failed_mols = []
            all_mols =[]
            wrong_structure=[]
            wrong_smiles=[]
            bad_index=[]
            for i, m in enumerate(supplier):
                structure = Chem.Mol(m)
                all_mols.append(structure)
                try:
                    Chem.SanitizeMol(structure)
                except:
                    failed_mols.append(m)
                    wrong_smiles.append(Chem.MolToSmiles(m))
                    wrong_structure.append(str(i+1))
                    bad_index.append(i)

           
            st.write('Original data: ', len(all_mols), 'molecules')
            # st.write('Kept data: ', len(moldf), 'molecules')
            st.write('Failed data: ', len(failed_mols), 'molecules')
            if len(failed_mols)!=0:
                number =[]
                for i in range(len(failed_mols)):
                    number.append(str(i+1))
                
                
                bad_molecules = pd.DataFrame({'No. failed molecule in original set': wrong_structure, 'SMILES of wrong structure: ': wrong_smiles, 'No.': number}, index=None)
                bad_molecules = bad_molecules.set_index('No.')
                st.dataframe(bad_molecules)

            # Standardization SDF file
            all_mols[:] = [x for i,x in enumerate(all_mols) if i not in bad_index] 
            records = []
            for i in range(len(all_mols)):
                record = Chem.MolToSmiles(all_mols[i])
                canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(record),isomericSmiles = False)
                records.append(canon_smi)
            
            moldf_n = []
            inchi_set=[]
            for i,record in enumerate(records):
                standard_record = standardize_smiles(record)
                m = Chem.MolFromSmiles(standard_record)
                moldf_n.append(m)
                inchi = str(Chem.MolToInchi(m))
                inchi_set.append(inchi)
           
            st.write('Kept data: ', len(moldf_n), 'molecules')

            # Calculate molecular descriptors
            def calcfp(mol,funcFPInfo=dict(radius=2,nBits=1024,useFeatures=False,useChirality = False)):
                arr = np.zeros((1,))
                fp = GetMorganFingerprintAsBitVect(mol, **funcFPInfo)
                DataStructs.ConvertToNumpyArray(fp, arr)
                return arr

            moldf=pd.DataFrame(moldf_n)
            moldf['Descriptors'] = moldf[0].apply(calcfp)
            X = np.array(list(moldf['Descriptors'])).astype(int)
            
            moldf.drop(columns='Descriptors', inplace=True)
            ######################
            # Pre-built model
            ######################

            # Apply model to make predictions
            number=[]
            for i in range(len(moldf)):
                number.append(str(i+1))
                
               
            exp=[]
            std=[]
            chembl_id=[]
            y_pred_con=[]
            cpd_AD_vs=[]
            number =[]
            count=0
            struct=[]
            structures=[]

            for inc in inchi_set:
                if inc in res:                    
                    exp.append(str(res[inc][0]['pchembl_value_mean']))
                    std.append(round(res[inc][0]['pchembl_value_std'],4))
                    chembl_id.append(str(res[inc][0]['molecule_chembl_id']))
                    y_pred_con.append('see experimental value')
                    cpd_AD_vs.append('-')
                    count+=1         
                    number.append(count) 
                else:
                    m = Chem.MolFromInchi(inc)
                    # Calculate molecular descriptors
                    f_vs = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, useFeatures=False, useChirality=False)]
                    X = rdkit_numpy_convert(f_vs)
                    #Predict activity
                    prediction_GBM = load_model_GBM.predict(X)
                    prediction_GBM = np.array(prediction_GBM)
                    y_pred = np.where(prediction_GBM == 1, "Active", "Inactive")                                     
                    y_pred_con.append(y_pred[0])


                    # Estimination AD                                   
                    d_ECFP4 = {}                
                    for mol in Chem.SDMolSupplier("datasets/HDAC6_ws_kekule.sdf"):
                        mg = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)
                        for m in moldf_n:
                            if m is not None:
                                mg_ = AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True)
                                d_ECFP4.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(mg, mg_))

                    df_ECFP4 = pd.DataFrame.from_dict(d_ECFP4)
                    str_a = np.where(df_ECFP4.max() >= threshold, "Inside AD", "Outside AD")                   
                    cpd_AD_vs.append(str_a[0])
                    exp.append('-')
                    std.append('-')
                    chembl_id.append('-')
                    count+=1         
                    number.append(count)             


            #Print and download common results

            st.header('**2. RESULTS OF PREDICTION:**')


            pred_beta = pd.DataFrame({'SMILES': records, 'HDAC6 activity': y_pred_con,'Applicability domain (AD)': cpd_AD_vs, 'No.': number, 'Experimental value, pIC50': exp,'Standard deviation': std, 'Chemble ID': chembl_id}, index=None)
            predictions = pred_beta.set_index('No.')
            count_exp=len(predictions[predictions['HDAC6 activity']=='see experimental value'])
            count_active=len(predictions[predictions['HDAC6 activity']=='Active'])
            count_active_AD=len(predictions[(predictions['HDAC6 activity']=='Active') & (predictions['Applicability domain (AD)']=='Inside AD')])
            count_inactive=len(predictions[predictions['HDAC6 activity']=='Inactive'])
            count_inactive_AD=len(predictions[(predictions['HDAC6 activity']=='Inactive') & (predictions['Applicability domain (AD)']=='Inside AD')])
            st.write('The total number of compounds which have experimental values: ', count_exp)
            st.write('Total active molecules: ', count_active)
            st.write('Total active molecules included in AD: ', count_active_AD)
            st.write('Total inactive molecules : ', count_inactive)
            st.write('Total inactive molecules included in AD: ', count_inactive_AD)
            if st.button('Show results as table'):                       
                st.dataframe(predictions)     
                def convert_df(df):
                    return df.to_csv().encode('utf-8')  
                csv = convert_df(predictions)

                st.download_button(
                    label="Download results of prediction as CSV",
                    data=csv,
                    file_name='Results.csv',
                    mime='text/csv',
                )

            for i in range(len(moldf)):
                a= moldf[0]
                b=list(a)
            # Print results for each molecules
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked = False
            def callback():
                st.session_state.button_clicked=True

            if (st.button('Show results, bioavailability radar, map of fragments contribution and for each molecule separately', on_click=callback) or st.session_state.button_clicked):
                st.header('**Prediction results:**')

                items_on_page = st.slider('Select number of compounds on page', 1, 15, 3)
                def paginator(label, items, items_per_page=items_on_page, on_sidebar=False):
                              
                # Figure out where to display the paginator
                    if on_sidebar:
                        location = st.sidebar.empty()
                    else:
                        location = st.empty()

                    # Display a pagination selectbox in the specified location.
                    items = list(items)
                    n_pages = len(items)
                    n_pages = (len(items) - 1) // items_per_page + 1
                    page_format_func = lambda i: "Page " + str(i+1)
                    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

                    # Iterate over the items in the page to let the user display them.
                    min_index = page_number * items_per_page
                    max_index = min_index + items_per_page
                    import itertools
                    return itertools.islice(enumerate(items), min_index, max_index)

                for i, m in paginator("Select a page", b):
                    smi = Chem.MolToSmiles(b[i])
                    m=b[i]
                    im = Draw.MolToImage(m)
                    st.write('**COMPOUNDS NUMBER **' + str(i+1) + '**:**')
                    st.write('**2D structure of compound number **' + str(i+1) + '**:**')
                    st.image(im)
                    # Lipinski's rule
                    st.header("**The Bioavailability Radar: сompliance the Lipinski's rule of five**")
                    lipinski(smi)

                    # 3D structure
                    st.write('**3D structure of compound number **'+ str(i+1) + '**:**')
                    blk=makeblock(smi)
                    render_mol(blk)
                    st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')

                    predictions = pd.DataFrame({'No. compound': i+1,'SMILES': smi, 'HDAC6 activity': y_pred_con[i],'Applicability domain (AD)': cpd_AD_vs[i], 'Experimental value, pIC50': exp[i],'Standard deviation': std[i], 'Chemble ID': chembl_id[i]}, index=[0])
                    
                    # CSS to inject contained in a string
                    hide_table_row_index = """
                                <style>
                                tbody th {display:none}
                                .blank {display:none}
                                </style>
                                """

                    # Inject CSS with Markdown
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)
                   
                    st.table(predictions)           

                    if y_pred_con[i]!='see experimental value':
                        st.write('**Predicted fragments contribution for compound number **'+ str(i+1) + '**:**')
                        fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(m, fpFunction, lambda x: getProba(x, load_model_GBM.predict_proba), colorMap=cm.PiYG_r)
                        st.pyplot(fig)
                        st.write('The chemical fragments are colored in green (predicted to reduce inhibitory activity) or magenta (predicted to increase activity HDAC6 inhibitors). The gray isolines separate positive and negative contributions.')
                        st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

if models_option == 'MLP_Topological fingerprints':
    load_model_MLP = pickle.load(open('Topological_FP/HDAC6_mlp_TFP.pkl', 'rb'))
    st.sidebar.header('Select input molecular files')
    threshold = 0.4
    def fpFunction(mol, atomId=-1):
            fp = SimilarityMaps.GetRDKFingerprint(mol,atomId=atomId)
            return fp
    # Read SMILES input
    SMILES = st.sidebar.checkbox('SMILES notations (*.smi)')    
    if SMILES:
        SMILES_input = ""
        compound_smiles = st.sidebar.text_area("Enter SMILES", SMILES_input)
        if len(compound_smiles)!=0:
            smiles=standardize_smiles(compound_smiles)
            m = Chem.MolFromSmiles(smiles)
            inchi = str(Chem.MolToInchi(m))
            im = Draw.MolToImage(m)
            st.sidebar.image(im)    
                
        if st.sidebar.button('PREDICT COMPOUND FROM SMILES'):
            # Calculate molecular descriptors
            
            f_vs=[Chem.RDKFingerprint(m)]
            X = rdkit_numpy_convert(f_vs)

            ######################
            # Pre-built model
            ######################

            # Apply model to make predictions
            prediction_MLP = load_model_MLP.predict(X)
            prediction_MLP = np.array(prediction_MLP)
            prediction_MLP = np.where(prediction_MLP == 1, "Active", "Inactive")


            # Estimination AD
            
            mol = Chem.MolFromSmiles(smiles)
            tp = Chem.RDKFingerprint(mol)

            d = {}
            for m in Chem.SDMolSupplier('datasets/HDAC6_ws.sdf'):
                if m is not None:
                    tp_ = Chem.RDKFingerprint(m)
                    d.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(tp, tp_))
            df_TPFP = pd.DataFrame.from_dict(d).T
            if df_TPFP[0].max()>=threshold:
                cpd_AD_vs = "Inside AD"
            else:
                cpd_AD_vs = "Outside AD"

            # search experimental value
            if inchi in res:
                exp=round(res[inchi][0]['pchembl_value_mean'],2)           
                std=round(res[inchi][0]['pchembl_value_std'],4)
                chembl_id=str(res[inchi][0]['molecule_chembl_id'])
                y_pred_con='see experimental value'
                cpd_AD_vs='-'
                
            else:
                y_pred_con=prediction_MLP[0]
                cpd_AD_vs=cpd_AD_vs
                exp="-"
                std="-"
                chembl_id="not detected"
                # Generate maps of fragment contribution
            
                fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(mol, fpFunction, lambda x: getProba(x, load_model_MLP.predict_proba), colorMap=cm.PiYG_r)
                st.write('**Predicted fragments contribution:**')
                st.pyplot(fig)
                st.write('The chemical fragments are colored in green (predicted to reduce inhibitory activity) or magenta (predicted to increase activity HDAC6 inhibitors). The gray isolines separate positive and negative contributions.')

            st.header('**Prediction results:**')


            common_inf = pd.DataFrame({'SMILES':smiles, 'Predicted value, pIC50': y_pred_con, 'Applicability domain': cpd_AD_vs,'Experimental value, pIC50': exp,'Standard deviation': std,
            'Chemble ID': chembl_id}, index=[1])
            predictions_pred=common_inf.astype(str) 
            st.dataframe(predictions_pred)                
            
            # Lipinski's rule
            st.header("**The Bioavailability Radar: сompliance the Lipinski's rule of five**")
            lipinski(compound_smiles)
            # 3D structure
            st.header('**3D structure of the studied compound:**')

            blk=makeblock(compound_smiles)
            render_mol(blk)
            st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')
    # Read SDF file 
    LOAD = st.sidebar.checkbox('MDL multiple SD file (*.sdf)')
    if LOAD:
        uploaded_file = st.sidebar.file_uploader("Choose a file")
        if uploaded_file is not None:
            st.header('**1. CHEMICAL STRUCTURE VALIDATION AND STANDARDIZATION:**')
            supplier = Chem.ForwardSDMolSupplier(uploaded_file,sanitize=False)
            failed_mols = []
            all_mols =[]
            wrong_structure=[]
            wrong_smiles=[]
            bad_index=[]
            for i, m in enumerate(supplier):
                structure = Chem.Mol(m)
                all_mols.append(structure)
                try:
                    Chem.SanitizeMol(structure)
                except:
                    failed_mols.append(m)
                    wrong_smiles.append(Chem.MolToSmiles(m))
                    wrong_structure.append(str(i+1))
                    bad_index.append(i)

           
            st.write('Original data: ', len(all_mols), 'molecules')
            st.write('Failed data: ', len(failed_mols), 'molecules')
            if len(failed_mols)!=0:
                number =[]
                for i in range(len(failed_mols)):
                    number.append(str(i+1))
                
                
                bad_molecules = pd.DataFrame({'No. failed molecule in original set': wrong_structure, 'SMILES of wrong structure: ': wrong_smiles, 'No.': number}, index=None)
                bad_molecules = bad_molecules.set_index('No.')
                st.dataframe(bad_molecules)

            # Standardization SDF file
            all_mols[:] = [x for i,x in enumerate(all_mols) if i not in bad_index] 
            records = []
            for i in range(len(all_mols)):
                record = Chem.MolToSmiles(all_mols[i])
                canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(record),isomericSmiles = False)
                records.append(canon_smi)
            
            moldf_n = []
            inchi_set=[]
            for i,record in enumerate(records):
                standard_record = standardize_smiles(record)
                m = Chem.MolFromSmiles(standard_record)
                moldf_n.append(m)
                inchi = str(Chem.MolToInchi(m))
                inchi_set.append(inchi)
           
            st.write('Kept data: ', len(moldf_n), 'molecules')
            

             # Calculate molecular descriptors
            def calcfp(mol):
                arr = np.zeros((1,))
                fp = Chem.RDKFingerprint(mol)
                DataStructs.ConvertToNumpyArray(fp, arr)
                return arr

            moldf=pd.DataFrame(moldf_n)
            moldf['Descriptors'] = moldf[0].apply(calcfp)
            X = np.array(list(moldf['Descriptors'])).astype(int)
            
            moldf.drop(columns='Descriptors', inplace=True)

                
            ######################
            # Pre-built model
            ######################

            # Apply model to make predictions
            number=[]
            for i in range(len(moldf)):
                number.append(str(i+1))
                
               
            exp=[]
            std=[]
            chembl_id=[]
            y_pred_con=[]
            cpd_AD_vs=[]
            number =[]
            count=0
            struct=[]
            structures=[]

            for inc in inchi_set:
                if inc in res:                    
                    exp.append(str(res[inc][0]['pchembl_value_mean']))
                    std.append(round(res[inc][0]['pchembl_value_std'],4))
                    chembl_id.append(str(res[inc][0]['molecule_chembl_id']))
                    y_pred_con.append('see experimental value')
                    cpd_AD_vs.append('-')
                    count+=1         
                    number.append(count) 
                else:
                    m = Chem.MolFromInchi(inc)
                    # Calculate molecular descriptors
                    f_vs = [Chem.RDKFingerprint(m)]
                    X = rdkit_numpy_convert(f_vs)
                    #Predict activity
                    prediction_MLP = load_model_MLP.predict(X)
                    prediction_MLP = np.array(prediction_MLP)
                    y_pred = np.where(prediction_MLP == 1, "Active", "Inactive")                                     
                    y_pred_con.append(y_pred[0])

                    # Estimination AD                                  
                    d_ECFP4 = {}                
                    for mol in Chem.SDMolSupplier("datasets/HDAC6_ws_kekule.sdf"):
                        mg = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)
                        for m in moldf_n:
                            if m is not None:
                                mg_ = AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True)
                                d_ECFP4.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(mg, mg_))

                    df_ECFP4 = pd.DataFrame.from_dict(d_ECFP4)
                    str_a = np.where(df_ECFP4.max() >= threshold, "Inside AD", "Outside AD")                   
                    cpd_AD_vs.append(str_a[0])
                    exp.append('-')
                    std.append('-')
                    chembl_id.append('-')
                    count+=1         
                    number.append(count)             

            #Print and download common results

            st.header('**2. RESULTS OF PREDICTION:**')

            pred_beta = pd.DataFrame({'SMILES': records, 'HDAC6 activity': y_pred_con,'Applicability domain (AD)': cpd_AD_vs, 'No.': number, 'Experimental value, pIC50': exp,'Standard deviation': std, 'Chemble ID': chembl_id}, index=None)
            predictions = pred_beta.set_index('No.')
            count_exp=len(predictions[predictions['HDAC6 activity']=='see experimental value'])
            count_active=len(predictions[predictions['HDAC6 activity']=='Active'])
            count_active_AD=len(predictions[(predictions['HDAC6 activity']=='Active') & (predictions['Applicability domain (AD)']=='Inside AD')])
            count_inactive=len(predictions[predictions['HDAC6 activity']=='Inactive'])
            count_inactive_AD=len(predictions[(predictions['HDAC6 activity']=='Inactive') & (predictions['Applicability domain (AD)']=='Inside AD')])
            st.write('The total number of compounds which have experimental values: ', count_exp)
            st.write('Total active molecules: ', count_active)
            st.write('Total active molecules included in AD: ', count_active_AD)
            st.write('Total inactive molecules : ', count_inactive)
            st.write('Total inactive molecules included in AD: ', count_inactive_AD)
            if st.button('Show results as table'):                       
                st.dataframe(predictions)     
                def convert_df(df):
                    return df.to_csv().encode('utf-8')  
                csv = convert_df(predictions)

                st.download_button(
                    label="Download results of prediction as CSV",
                    data=csv,
                    file_name='Results.csv',
                    mime='text/csv',
                )

            for i in range(len(moldf)):
                a= moldf[0]
                b=list(a)
            
            
            # Print results for each molecules
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked = False
            def callback():
                st.session_state.button_clicked=True
            if (st.button('Show results, bioavailability radar, map of fragments contribution and for each molecule separately', on_click=callback) or st.session_state.button_clicked):
                st.header('**Prediction results:**')

                items_on_page = st.slider('Select number of compounds on page', 1, 15, 3)
                def paginator(label, items, items_per_page=items_on_page, on_sidebar=False):
                              
                # Figure out where to display the paginator
                    if on_sidebar:
                        location = st.sidebar.empty()
                    else:
                        location = st.empty()

                    # Display a pagination selectbox in the specified location.
                    items = list(items)
                    n_pages = len(items)
                    n_pages = (len(items) - 1) // items_per_page + 1
                    page_format_func = lambda i: "Page " + str(i+1)
                    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

                    # Iterate over the items in the page to let the user display them.
                    min_index = page_number * items_per_page
                    max_index = min_index + items_per_page
                    import itertools
                    return itertools.islice(enumerate(items), min_index, max_index)

                for i, m in paginator("Select a page", b):
                    smi = Chem.MolToSmiles(b[i])
                    m=b[i]
                    im = Draw.MolToImage(m)
                    st.write('**COMPOUNDS NUMBER **' + str(i+1) + '**:**')
                    st.write('**2D structure of compound number **' + str(i+1) + '**:**')
                    st.image(im)
                    # Lipinski's rule
                    st.header("**The Bioavailability Radar: сompliance the Lipinski's rule of five**")
                    lipinski(smi)
                    # 3D structure
                    st.write('**3D structure of compound number **'+ str(i+1) + '**:**')
                    blk=makeblock(smi)
                    render_mol(blk)
                    st.write('You can use the scroll wheel on your mouse to zoom in or out a 3D structure of compound')

                    predictions = pd.DataFrame({'No. compound': i+1,'SMILES': smi, 'HDAC6 activity': y_pred_con[i],'Applicability domain (AD)': cpd_AD_vs[i], 'Experimental value, pIC50': exp[i],'Standard deviation': std[i], 'Chemble ID': chembl_id[i]}, index=[0])
                    
                    # CSS to inject contained in a string
                    hide_table_row_index = """
                                <style>
                                tbody th {display:none}
                                .blank {display:none}
                                </style>
                                """

                    # Inject CSS with Markdown
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)
                   
                    st.table(predictions)           

                    if y_pred_con[i]!='see experimental value':
                        st.write('**Predicted fragments contribution for compound number **'+ str(i+1) + '**:**')
                        def fpFunction(m, atomId=-1):
                            fp = SimilarityMaps.GetRDKFingerprint(m,atomId=atomId)
                            return fp

                        fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(m, fpFunction, lambda x: getProba(x, load_model_MLP.predict_proba), colorMap=cm.PiYG_r)
                        st.pyplot(fig)
                        st.write('The chemical fragments are colored in green (predicted to reduce inhibitory activity) or magenta (predicted to increase activity HDAC6 inhibitors). The gray isolines separate positive and negative contributions.')
                        st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    
st.text('© Oleg Tinkov, 2022')      
