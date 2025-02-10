# utils.py
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.savefig(f'relatorio/graficos/cm_{model_name}.png')
    plt.close()

# Exemplo de uso:
for name, data in results.items():
    plot_confusion_matrix(data["CM"], name)