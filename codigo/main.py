# main.py
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import unicodedata
import re

# ==================== CONFIGURAÇÕES INICIAIS ====================
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8')
sns.set_theme(style='whitegrid', palette='viridis')

# Caminho relativo à planilha
caminho_relativo = os.path.join('..', 'dados', 'Pedido_informacao_sei_23064.030111_2019_10_Versao02.xlsx')

# ==================== CARREGAMENTO DE DADOS ====================
def carregar_dados():
    print("\n[1/10] Carregando dados...")
    df_relacao = pd.read_excel(caminho_relativo, sheet_name='Relação de Alunos')
    df_historico = pd.read_excel(caminho_relativo, sheet_name='Historico')
    df_questionario = pd.read_excel(caminho_relativo, sheet_name='Questionario_socioEconomico')
    
    print("\n[DEBUG] Colunas nas abas:")
    print("- Relação de Alunos:", df_relacao.columns.tolist())
    print("- Questionário:", df_questionario.columns.tolist())
    print("- Histórico:", df_historico.columns.tolist())
    
    return df_relacao, df_historico, df_questionario

# ==================== ANÁLISE DESCRITIVA ====================
def analise_descritiva(df):
    """
    Gera análises descritivas e visualizações iniciais dos dados.
    """
    print("\n[2/10] Análise descritiva dos dados...")
    
    # Estatísticas descritivas
    print("\nEstatísticas descritivas:")
    print(df.describe())
    
    # Codificar a variável target (Situação Atual do Aluno)
    le = LabelEncoder()
    df['Situação Atual do Aluno'] = le.fit_transform(df['Situação Atual do Aluno'])
    
    # Distribuição da variável target (evasão)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Situação Atual do Aluno', data=df)
    plt.title('Distribuição da Evasão', fontsize=14)
    plt.xlabel('Evasão (0 = Não Evadiu, 1 = Evadiu)', fontsize=12)
    plt.ylabel('Contagem', fontsize=12)
    plt.tight_layout()
    plt.savefig('relatorio/graficos/distribuicao_evasao.png')
    plt.close()
    
    # Correlação entre variáveis numéricas
    plt.figure(figsize=(12, 8))
    corr_matrix = df[['Coeficiente', 'Nota Enem', 'Idade', 'Freq.(%)', 'Média da Turma', 'Situação Atual do Aluno']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlação entre Variáveis Numéricas', fontsize=14)
    plt.tight_layout()
    plt.savefig('relatorio/graficos/correlacao_variaveis.png')
    plt.close()

# ==================== PRÉ-PROCESSAMENTO ====================
def pre_processamento(df_relacao, df_historico, df_questionario):
    """
    Realiza o pré-processamento dos dados.
    """
    print("\n[3/10] Pré-processamento dos dados...")
    
    # Unir dados
    df = pd.merge(df_relacao, df_questionario, on='id', how='left')
    df = pd.merge(df, df_historico, on='id', how='left')
    
    # Codificar variável target
    le = LabelEncoder()
    df['Situação Atual do Aluno'] = le.fit_transform(df['Situação Atual do Aluno'])
    
    # Selecionar features e target
    features = ['Coeficiente', 'Nota Enem', 'Sexo', 'Idade', 'Freq.(%)', 'Média da Turma']
    target = 'Situação Atual do Aluno'
    
    X = df[features].copy()
    y = df[target]
    
    # Codificar variáveis categóricas
    X = pd.get_dummies(X, columns=['Sexo'], drop_first=True)
    
    # Tratar valores faltantes
    numeric_cols = X.select_dtypes(include=np.number).columns
    imputer = SimpleImputer(strategy='mean')
    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
    
    # Normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, df, X.columns

# ==================== ANÁLISE DE EVASÃO ====================
def analise_evasao(df):
    """
    Gera gráficos e análises específicas sobre a evasão dos alunos.
    """
    print("\n[4/10] Análise de evasão...")
    
    # Taxa de evasão por período
    evasao_periodo = df.groupby('Período do Aluno')['Situação Atual do Aluno'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Período do Aluno', y='Situação Atual do Aluno', data=evasao_periodo)
    plt.title('Taxa de Evasão por Período', fontsize=14)
    plt.xlabel('Período', fontsize=12)
    plt.ylabel('Taxa de Evasão', fontsize=12)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('relatorio/graficos/evasao_periodo.png')
    plt.close()

    # Exibir resultados
    print("\nTaxa de Evasão por Período:")
    print(evasao_periodo)

    # Função para normalizar nomes de disciplinas (incluindo tratamento de números romanos)
    def normalizar_nome_disciplina(nome):
        nome = str(nome)
        # Substituir números romanos por arábicos (ex: "I" → "1")
        roman_to_arabic = {" I": " 1", " II": " 2", " III": " 3", " IV": " 4", " V": " 5"}
        for roman, arabic in roman_to_arabic.items():
            nome = re.sub(re.escape(roman) + r'\b', arabic, nome, flags=re.IGNORECASE)
        # Normalização padrão
        nome = unicodedata.normalize('NFKD', nome).encode('ASCII', 'ignore').decode('ASCII')
        nome = re.sub(r'[^a-zA-Z0-9\s]', '', nome)
        nome = re.sub(r'\s+', ' ', nome)
        return nome.strip().upper()

    # Aplicar normalização
    df['Nome Disciplina'] = df['Nome Disciplina'].apply(normalizar_nome_disciplina)
    df['Cod. Disciplina'] = df['Cod. Disciplina'].astype(str).str.strip().str.upper()

    # Remover registros duplicados (mesmo aluno na mesma disciplina)
    df = df.drop_duplicates(subset=['id', 'Cod. Disciplina'])

    # Definir situações de evasão
    situacoes_evasao = ["Reprovado por Nota/Frequência", "Reprovado por Nota", "Reprovado por Frequência", "Reprovado em Exame de Suficiência", "Reprovado", "Cancelado"]
    df['Evasao'] = df['Situação Disc.'].apply(lambda x: 1 if x in situacoes_evasao else 0)

    # Agrupar por código e nome da disciplina
    evasao_materia = (
        df.groupby(['Cod. Disciplina', 'Nome Disciplina'])
        .agg(Total_Alunos=('id', 'count'), Total_Evasões=('Evasao', 'sum'))
        .reset_index()
    )

    # Filtrar turmas com pelo menos 10 alunos
    evasao_materia = evasao_materia[evasao_materia['Total_Alunos'] >= 10]
    print(f"Número de disciplinas após o filtro: {len(evasao_materia)}")

    # Calcular taxa de evasão
    evasao_materia['Taxa de Evasão'] = evasao_materia['Total_Evasões'] / evasao_materia['Total_Alunos']
    
    # Agrupar por nome da disciplina e calcular a taxa de evasão agrupada
    evasao_agrupada = (
        evasao_materia.groupby('Nome Disciplina')
        .agg(Total_Alunos=('Total_Alunos', 'sum'), Total_Evasões=('Total_Evasões', 'sum'))
        .reset_index()
    )
    
    # Calcular a taxa de evasão agrupada
    evasao_agrupada['Taxa de Evasão'] = evasao_agrupada['Total_Evasões'] / evasao_agrupada['Total_Alunos']
    
    # Ordenar e selecionar as top 10 matérias
    top_10_agrupado = evasao_agrupada.sort_values(by='Taxa de Evasão', ascending=False).head(10)
    
    # Plotar gráfico
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Taxa de Evasão', y='Nome Disciplina', data=top_10_agrupado)
    plt.title('Top 10 Matérias com Maior Taxa de Evasão', fontsize=14)
    plt.xlabel('Taxa de Evasão', fontsize=12)
    plt.ylabel('Matéria', fontsize=12)
    plt.tight_layout()
    plt.savefig('relatorio/graficos/evasao_materia.png')
    plt.close()

    # Exibir resultados
    print("Top 10 Matérias com Maior Taxa de Evasão (Agrupadas por Nome):")
    print(top_10_agrupado[['Nome Disciplina', 'Total_Alunos', 'Taxa de Evasão']])

# ==================== MODELAGEM E AVALIAÇÃO ====================
def modelagem_e_avaliacao(X_scaled, y, feature_names):
    """
    Treina e avalia os modelos de machine learning.
    """
    print("\n[5/10] Modelagem e avaliação...")
    
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Modelos
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_jobs=-1),
        "Linear SVM": LinearSVC(class_weight='balanced', max_iter=10000),
        "KNN": KNeighborsClassifier()
    }
    
    resultados = []
    for name, model in models.items():
        try:
            print(f"\nTreinando {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            
            resultados.append({
                'Modelo': name,
                'Acurácia': accuracy,
                'Matriz_Confusão': cm,
                'Relatório': report
            })
            
            # Importância das variáveis (se aplicável)
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 6))
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.title(f"Importância das Variáveis - {name}", fontsize=14)
                plt.barh(range(len(indices)), importances[indices], align='center')
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Importância Relativa', fontsize=12)
                plt.xlim(0, 1)
                plt.tight_layout()
                plt.savefig(f'relatorio/graficos/feature_importance_{name}.png')
                plt.close()
                
        except Exception as e:
            print(f"\n[ERRO] {name}: {str(e)}")
    
    return resultados

# ==================== COMPARAÇÃO DE MODELOS ====================
def comparacao_modelos(resultados):
    """
    Compara o desempenho dos modelos e gera visualizações.
    """
    print("\n[6/10] Comparação de modelos...")
    
    # Criar DataFrame com os resultados
    df_resultados = pd.DataFrame(resultados)
    
    # Gráfico de acurácia
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Modelo', y='Acurácia', data=df_resultados)
    plt.title('Acurácia dos Modelos', fontsize=14)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('relatorio/graficos/acuracia_modelos.png')
    plt.close()
    
    # Salvar resultados em CSV
    df_resultados.to_csv('relatorio/resultados_modelos.csv', index=False)

# ==================== EXPORTAÇÃO DE RESULTADOS ====================
def exportar_resultados(resultados):
    """
    Exporta os resultados para arquivos.
    """
    print("\n[7/10] Exportando resultados...")
    
    # Salvar resultados dos modelos
    pd.DataFrame(resultados).to_csv('relatorio/resultados_modelos.csv', index=False)
    
    # Salvar gráficos e análises
    print("Gráficos e análises salvos na pasta 'relatorio/graficos/'.")

# ==================== EXECUÇÃO PRINCIPAL ====================
if __name__ == "__main__":
    # Etapa Exploratória
    print("\n[ETAPA EXPLORATÓRIA]")
    df_relacao, df_historico, df_questionario = carregar_dados()
    df = pd.merge(df_relacao, df_questionario, on='id', how='left')
    df = pd.merge(df, df_historico, on='id', how='left')
    analise_descritiva(df)  # Gráficos descritivos
    analise_evasao(df)      # Gráficos de evasão

    # Etapa Preditiva
    print("\n[ETAPA PREDITIVA]")
    X_scaled, y, df, feature_names = pre_processamento(df_relacao, df_historico, df_questionario)
    resultados = modelagem_e_avaliacao(X_scaled, y, feature_names)  # Treina e avalia modelos
    comparacao_modelos(resultados)  # Compara modelos
    exportar_resultados(resultados)  # Exporta resultados

    print("\n[CONCLUSÃO] Processo concluído! Verifique os arquivos na pasta 'relatorio'.")