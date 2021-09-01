import pandas as pd
from tqdm.notebook import tqdm


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