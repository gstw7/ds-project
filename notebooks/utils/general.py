import numpy as np
import pandas as pd
from boruta import BorutaPy
from tqdm.notebook import tqdm
from sklearn.pipeline import Pipeline
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier


def get_data_abono(ano):

    columns = [
    'NOME', 
    'CPF', 
    'DSC_CARGO_EMPREGO', 
    'NV_ESCOLARIDADE', 
    'DN_ORGAO_ATUACAO', 
    'UF_UPAG', 
    'DN_UNIDADE_ORGANIZACIONAL', 
    'UF_RESIDENCIA', 
    'CIDADE_RESIDENCIA', 
    'SITUACAO_SVR', 
    'QT_ANOS_SVR_PUBLICOS', 
    'QT_MESES_SVR_PUBLICOS', 
    'ANO-MES_INICIO_ABONO_PERMANE', 
    'VAL'
    ]

    results = pd.DataFrame(columns=columns)
    for j in tqdm(ano): 
        for i in range(1, 13):
            if i < 10:
                try:
                    url = f'http://repositorio.dados.gov.br/segrt/ABONOP_0{i}{j}.csv'
                    df = pd.read_csv(url, encoding='ISO-8859-1', sep=';', names=columns)[1:]
                except:
                    continue
            if i >= 10:
                try:
                    url = f'http://repositorio.dados.gov.br/segrt/ABONOP_{i}{j}.csv'
                    df = pd.read_csv(url, encoding='ISO-8859-1', sep=';', names=columns)[1:]
                except:
                    continue
            results = pd.concat([results, df])
    return results.to_csv('../data/raw/abono.csv', sep=';', index=False)


def get_data_aposentado(ano):

    columns = [
    'NOME',
    'CPF',
    'MAT_SERVIDOR',
    'NM_ORGAO',
    'SIGLA_ORGAO',
    'CD_ORGAO',
    'CARGO',
    'CLASSE',
    'PADRAO',
    'REF',
    'NIVEL',
    'TP_APOSENTADORIA',
    'FUND_INATIVIDADE',
    'NM_DIPLO_LEGAL',
    'DT_PUBLI_DO_DL',
    'OC_INGRESSO_SVP',
    'DT_OC_INGRESSO_SVP',
    'VL_RENDIMENTO_LIQUIDO'
    ] 

    results = pd.DataFrame(columns=columns)
    for j in tqdm(ano): 
        for i in range(1, 13):
            if i < 10:
                try:
                    
                    url = f'http://repositorio.dados.gov.br/segrt/APOSENTADOS_0{i}{j}.csv'
                    df = pd.read_csv(url, encoding='ISO-8859-1', sep=';', header=None, names=columns)
                except:
                    continue
            if i >= 10:
                try:
                    url = f'http://repositorio.dados.gov.br/segrt/APOSENTADOS_{i}{j}.csv'
                    df = pd.read_csv(url, encoding='ISO-8859-1', sep=';', header=None, names=columns)
                except:
                    continue
            results = pd.concat([results, df])
    return results.to_csv('../data/raw/aposentados.csv', sep=';', index=False)

def aux(df):

    df_aux = pd.DataFrame(
        {
            'colunas' : df.columns,
            'tipo': df.dtypes,
            'missing' : df.isna().sum(),
            'size' : df.shape[0],
            'unicos': df.nunique()
            }
        )
    df_aux['percentual%'] = round(df_aux['missing'] / df_aux['size'], 3)*100
    
    return df_aux

def get_date(date):
    
    size = len(str(date))
    if size == 7:
        date = '0'+str(date)
    else:
        date = date
    return date

def remove_outlier_IQR(df):
    
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
    return df_final

def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

def chi_squared(df, y, cols = None):
    pvalues = []
    logs = []
    chi2_list = []
    if cols == None:
        cat_columns = df.select_dtypes(['object']).columns.tolist()
    else:
        cat_columns = cols
    for cat in tqdm(cat_columns):
        table = pd.crosstab(df[cat], df[y])
        if not table[table < 5 ].count().any():
            table = pd.crosstab(df[cat], df[y])
            chi2, p, dof, expected = chi2_contingency(table.values) # Função que realiza o teste
            chi2_list.append(chi2)
            pvalues.append(p)
        else:
            logs.append("A coluna {} não pode ser avaliada. ".format(cat))
            chi2_list.append(np.nan)
            pvalues.append(np.nan)   
    chi2_df = pd.DataFrame({"column":cat_columns, 'p-value':pvalues, 'chi2_value':chi2_list})
    return  chi2_df, logs


def boruta_selector(df, y=None):
    
    SEED = 1
    
    Y = df[y]
    df = df.drop(y,axis=1)
    
    num_feat = df.select_dtypes(include=['int','float']).columns.tolist()
    cat_feat = df.select_dtypes(include=['object']).columns.tolist()
    
    pipe_num_tree = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
    pipe_cat_tree = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', OrdinalEncoder())])
    preprocessor_tree = ColumnTransformer( transformers = [('num_preprocessor',pipe_num_tree, num_feat), ('cat_preprocessor', pipe_cat_tree, cat_feat)])
    
    X = preprocessor_tree.fit_transform(df)
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)    

    feat_selector = BorutaPy(rf, n_estimators='auto', random_state=SEED, max_iter = 100) # 500 iterações até convergir
    feat_selector.fit(X,Y)
    # Terceiro filtro com as features selecionadas pelo boruta
    cols_drop_boruta= [not x for x in feat_selector.support_.tolist()] # apenas invertendo o vetor de true/false
    cols_drop_boruta= df.loc[:, cols_drop_boruta].columns.tolist()
    return cols_drop_boruta
