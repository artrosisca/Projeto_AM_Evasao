# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carregar dados das abas do Excel
df_relacao = pd.read_excel('Book1.xlsx', sheet_name='Relação de Alunos')
df_historico = pd.read_excel('Book1.xlsx', sheet_name='Histórico')
df_questionario = pd.read_excel('Book1.xlsx', sheet_name='Questionario_socioEconomico')

# Unir dados pelo 'id' (chave primária)
df = pd.merge(df_relacao, df_questionario, on='id', how='left')
df = pd.merge(df, df_historico, on='id', how='left')

# Excluir registros sem questionário socioeconômico (sua escolha)
df = df.dropna(subset=df_questionario.columns.difference(['id']))

# Codificar variáveis categóricas (exemplo: 'Situação atual do aluno')
le = LabelEncoder()
df['Situação atual do aluno'] = le.fit_transform(df['Situação atual do aluno'])  # 0 = Evadido, 1 = Ativo

# Selecionar features e target
features = ['Coeficiente', 'Escore Vest', 'Nota Enem', 'Sexo', 'Idade', 'Freq.(%)', 'Média da Turma']
target = 'Situação atual do aluno'

# Pré-processamento
X = df[features]
y = df[target]

# Codificar variáveis categóricas restantes (ex: Sexo)
X = pd.get_dummies(X, columns=['Sexo'])

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Lista de modelos
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()  # Bônus para comparação
}

# Treinar e avaliar modelos
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = {"Accuracy": accuracy, "CM": cm, "Report": report}

# Salvar resultados em tabela
results_df = pd.DataFrame({k: [v["Accuracy"]] for k, v in results.items()})
results_df.to_csv('resultados_modelos.csv', index=False)