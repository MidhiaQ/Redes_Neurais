"""
Classificação de Desempenho Estudantil com Redes Neurais Artificiais (MLP)
Dataset: xAPI-Edu-Data (Kaggle)
Análise comparativa por subgrupos de gênero (Feminino vs. Masculino)
"""

import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Configurações de reprodutibilidade
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Diretório para salvar gráficos e métricas
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def carregar_dados() -> pd.DataFrame:
    """
    Carrega o dataset xAPI-Edu-Data via kagglehub ou arquivo local.
    """
    local_file = "xAPI-Edu-Data.csv"

    if os.path.exists(local_file):
        print(f"[INFO] Carregando dataset a partir do arquivo local: '{local_file}'...")
        df = pd.read_csv(local_file)
    else:
        print("[INFO] Baixando dataset 'aljarah/xAPI-Edu-Data' via kagglehub...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "aljarah/xAPI-Edu-Data",
            local_file,
        )

    # Tradução das classes para português
    traducao_classes = {
        'H': 'Alto',
        'M': 'Medio',
        'L': 'Baixo (Risco)'
    }
    df['Class'] = df['Class'].map(traducao_classes)

    print(f"[INFO] Dados carregados com sucesso! Total de registros: {len(df)}")
    return df


def criar_modelo_mlp(num_atributos: int, num_classes: int) -> tf.keras.Model:
    """
    Cria a arquitetura da Rede Neural Perceptron Multicamadas (MLP).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_atributos,)),
        tf.keras.layers.Dense(16, activation='relu', name='camada_oculta_1'),
        tf.keras.layers.Dense(8, activation='relu', name='camada_oculta_2'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='camada_saida')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def treinar_avaliar_mlp(df_subgrupo: pd.DataFrame, nome_grupo: str):
    """
    Executa o pré-processamento, treinamento e avaliação da MLP para um subgrupo específico.
    """
    print(f"\n{'='*60}")
    print(f" TREINANDO MLP PARA O GRUPO: {nome_grupo.upper()}")
    print(f"{'='*60}")

    # Separar atributos preditores e o alvo
    # Removemos 'Class' (alvo) e 'gender' (variável de estratificação do grupo)
    X_bruto = df_subgrupo.drop(columns=['Class', 'gender'])
    y_bruto = df_subgrupo['Class'].values.reshape(-1, 1)

    # Converter variáveis categóricas em numéricas (One-Hot Encoding nas entradas)
    X = pd.get_dummies(X_bruto, drop_first=True).values.astype(np.float32)

    # Codificar a variável-alvo multiclasse (One-Hot Encoding nas saídas)
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y_bruto)
    nomes_classes = encoder.categories_[0]

    # Divisão em Treino e Teste (80% treino / 20% teste)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Normalização dos Dados (Z-score)
    padronizador = StandardScaler()
    X_treino = padronizador.fit_transform(X_treino)
    X_teste = padronizador.transform(X_teste)

    # Construção da MLP
    num_atributos = X_treino.shape[1]
    num_classes = y.shape[1]
    model = criar_modelo_mlp(num_atributos, num_classes)

    # Treinamento da Rede Neural
    historico = model.fit(
        X_treino, y_treino,
        epochs=60,
        batch_size=16,
        validation_split=0.1,
        verbose=0
    )

    # Visualização e Salvamento do Histórico de Aprendizagem (Loss)
    plt.figure(figsize=(6, 4))
    plt.plot(historico.history['loss'], label='Treino', linewidth=2)
    plt.plot(historico.history['val_loss'], label='Validação', linewidth=2)
    plt.title(f'Histórico de Aprendizagem (Loss) - Grupo {nome_grupo}')
    plt.ylabel('Loss (Erro)')
    plt.xlabel('Épocas')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    loss_path = os.path.join(OUTPUT_DIR, f'curva_loss_{nome_grupo.lower()}.png')
    plt.savefig(loss_path, dpi=300)
    plt.show()
    print(f"[ARTEFATO] Gráfico de perda salvo em: {loss_path}")

    # Avaliação no Teste
    erro_teste, acuracia_teste = model.evaluate(X_teste, y_teste, verbose=0)
    print(f"\nAcurácia (Accuracy) no Teste: {acuracia_teste * 100:.2f}%\n")

    # Predições e Classes
    predicoes = model.predict(X_teste, verbose=0)
    pred_classes = np.argmax(predicoes, axis=1)
    y_teste_classes = np.argmax(y_teste, axis=1)

    # Relatório de Classificação (Precision, Recall, F1-Score)
    print(f"--- MÉTRICAS DE AVALIAÇÃO ({nome_grupo.upper()}) ---")
    relatorio = classification_report(y_teste_classes, pred_classes, target_names=nomes_classes)
    print(relatorio)

    # Salvar relatório em texto
    report_path = os.path.join(OUTPUT_DIR, f'relatorio_{nome_grupo.lower()}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Métricas de Avaliação - Grupo {nome_grupo}\n")
        f.write(f"Acurácia no Teste: {acuracia_teste * 100:.2f}%\n\n")
        f.write(relatorio)

    # Matriz de Confusão
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_teste_classes, pred_classes)
    cmap = 'Purples' if nome_grupo.lower() == 'feminino' else 'Blues'
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=nomes_classes,
        yticklabels=nomes_classes
    )
    plt.title(f'Matriz de Confusão - {nome_grupo}')
    plt.xlabel('Predição da MLP')
    plt.ylabel('Classe Real')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, f'matriz_confusao_{nome_grupo.lower()}.png')
    plt.savefig(cm_path, dpi=300)
    plt.show()
    print(f"[ARTEFATO] Matriz de confusão salva em: {cm_path}")

    return acuracia_teste


def main():
    # 1. Carregar os dados
    df = carregar_dados()

    # 2. Separação dos subgrupos por gênero
    df_feminino = df[df['gender'] == 'F'].copy()
    df_masculino = df[df['gender'] == 'M'].copy()

    print(f"[INFO] Amostras Grupo Feminino : {len(df_feminino)}")
    print(f"[INFO] Amostras Grupo Masculino: {len(df_masculino)}")

    # 3. Treinar e avaliar para ambos os subgrupos
    acc_fem = treinar_avaliar_mlp(df_feminino, "Feminino")
    acc_masc = treinar_avaliar_mlp(df_masculino, "Masculino")

    # 4. Resumo comparativo final
    print("\n" + "="*50)
    print("           RESUMO COMPARATIVO FINAL           ")
    print("="*50)
    print(f"Acurácia MLP Grupo Feminino : {acc_fem * 100:.2f}%")
    print(f"Acurácia MLP Grupo Masculino: {acc_masc * 100:.2f}%")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
