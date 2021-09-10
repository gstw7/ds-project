Organização do projeto
------------

    ├── LICENSE       
    ├── README.md          
    ├── data
    │   ├── processed
    │   ├── interim      
    │   ├── raw            
    │  
    ├── models            
    │
    ├── notebooks
    │    └── utils                               
    │  
    ├── figures  
    │
    ├── requirements.txt 
    │                         
    └──
--------

- [Dicioário das variáveis](https://github.com/gstw7/ds-project/blob/master/data/raw/README.md)
## Requisitos
- [Anaconda](https://www.anaconda.com/products/individual-d)

## Como rodar?
1. Instalar o Anaconda;
2. Abrir o terminal ou prompt de comando dentro da pasta do projeto e executar ```conda create -n <environment-name> --file requirements.txt```;
3. Executar ```conda activate <environment-name>```;
4. Executar ```jupyter notebook```;
5. Após abrir o Jupyter Notebook ir até a pasta notebooks e executar os notebooks (arquivos .ipynb) na seguinte ordem:
    - 01 - GET_DATA >> 02 - DATA_CLEANING >> 03 - EDA >> 04 - FEATURE_SELECT >> 05 - MODELING