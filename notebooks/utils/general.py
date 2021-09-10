import warnings
import numpy as np
import pandas as pd
import scikitplot as skplt
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from yellowbrick.classifier import (
    ClassificationReport, 
    ConfusionMatrix, 
    DiscriminationThreshold, 
    PrecisionRecallCurve, 
    ROCAUC
    )

warnings.filterwarnings('ignore')



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
            chi2, p, dof, expected = chi2_contingency(table.values)
            chi2_list.append(chi2)
            pvalues.append(p)
        else:
            logs.append("A coluna {} não pode ser avaliada. ".format(cat))
            chi2_list.append(np.nan)
            pvalues.append(np.nan)   
    chi2_df = pd.DataFrame({"column":cat_columns, 'p-value':pvalues, 'chi2_value':chi2_list})
    return  chi2_df, logs


def viz_performance(X_train, X_test, y_train, y_test, clf, classes, figsize=(12, 16), cmap='Greens'):

    fig, ax = plt.subplots(3, 2, figsize=figsize)
    
    lr = clf.fit(X_train, y_train)
    y_probas = lr.predict_proba(X_test)
    skplt.metrics.plot_ks_statistic(y_test, y_probas, ax=ax[2,1])
    
    grid = [
        ConfusionMatrix(clf, ax=ax[0, 0], classes=classes, cmap=cmap),
        ClassificationReport(clf, ax=ax[0, 1], classes=classes, cmap=cmap ),
        PrecisionRecallCurve(clf, ax=ax[1, 0]),
        ROCAUC(clf, ax=ax[1, 1], micro=False, macro=False, per_class=True, classes=classes),
        DiscriminationThreshold(clf, ax=ax[2,0])
    ]
    
    for viz in grid:
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()

    plt.tight_layout()
    plt.show()
