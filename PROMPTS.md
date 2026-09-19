# Registro de Prompts de Inteligência Artificial

Este documento reúne o registro e a rastreabilidade dos prompts de Inteligência Artificial utilizados durante as etapas de desenvolvimento, experimentação e escrita científica deste trabalho, atendendo às boas práticas de integridade acadêmica e transparência metodológica.

---

## Prompt 1: Tutoria e Adaptação do Pipeline de Machine Learning

- **Objetivo:** Orientar a evolução do código didático (Iris/MNIST) para a arquitetura de estratificação por subgrupos de gênero e tratamento de atributos categóricos no dataset educacional.
- **Área:** Aprendizado de Máquina, Pré-processamento de Dados e Engenharia de Software.
- **Finalidade:** Desenvolvimento técnico e modularização do pipeline.

### Conteúdo do Prompt:

```text
Atue como meu tutor de Machine Learning. Na aula, eu construí códigos básicos de classificação com Redes Neurais (MLP) baseados nos exemplos da Íris e do MNIST, onde treinamos um único modelo para o conjunto de dados inteiro.

Agora, vou adaptar a base desse código para o meu projeto, mas com um acréscimo central: preciso separar meu dataset educacional em dois subgrupos (Feminino e Masculino) e treinar um modelo para cada um, para poder compará-los.

Por favor, me ensine passo a passo como evoluir o código da aula para essa nova estrutura. Quero que você me guie nas seguintes etapas:

Como usar o Pandas para dividir o dataset original nesses dois grupos distintos.

Como tratar as variáveis categóricas para que a rede neural consiga processá-las.

Como criar uma função (def) para encapsular a montagem e o treinamento da rede, para que eu não precise copiar e colar o mesmo código duas vezes.

Regra fundamental: Não me dê o código pronto. Explique o primeiro passo de forma simples, mostre como começar a escrever.
```

---

## Prompt 2: Mentoria e Guia para Escrita do Artigo Científico (Formato IEEE)

- **Objetivo:** Estabelecer a metodologia de mentoria orientada para a redação do artigo científico de 4 páginas em formato IEEE (LaTeX).
- **Área:** Mineração de Dados Educacionais (Educational Data Mining - EDM) e Inteligência Artificial.
- **Finalidade:** Estruturação da comunicação científica e discussão de resultados.

### Conteúdo do Prompt:

```text
Atue como um mentor acadêmico sênior e pesquisador especialista em Inteligência Artificial e Mineração de Dados Educacionais (EDM). Meu objetivo é escrever um artigo científico de 4 páginas no formato IEEE Explore (em LaTeX), focado na classificação do desempenho acadêmico de estudantes usando Redes Neurais Perceptron Multicamadas (MLP), com uma análise comparativa entre os gêneros masculino e feminino.

DIRETRIZES PARA A NOSSA INTERAÇÃO:
1. NÃO escreva o artigo para mim. 
2. Quero que você me guie seção por seção (Introdução, Referencial Teórico, Metodologia, Resultados e Discussão, Conclusão).
3. Para cada seção, faça o seguinte:
   - Explique brevemente o que devo abordar com base nos resultados do meu código.
   - Forneça um pequeno esboço ou estrutura de parágrafos.
   - Peça para eu escrever a minha versão ou pergunte se quero que você sugira um primeiro rascunho para aquela seção específica.
4. Ao chegarmos na seção de Resultados, me oriente sobre onde e como criar os blocos LaTeX (\begin{figure}) para inserir as imagens geradas pelo código (curvas de loss, matrizes de confusão e gráficos de engajamento).
```

---

## Prompt 3: Revisão Textual, Gramatical e Polimento Científico

- **Objetivo:** Estabelecer o protocolo de revisão do texto acadêmico em duas etapas (correção gramatical estrita mantendo a voz autoral + sugestões de aprimoramento estilístico/formal).
- **Área:** Escrita Científica e Comunicação Técnica.
- **Finalidade:** Revisão por pares, correção ortográfica e elevação do registro acadêmico.

### Conteúdo do Prompt:

```text
Atue como um revisor profissional de textos acadêmicos e científicos. Abaixo, vou fornecer um trecho que escrevi para o meu artigo.

Sua tarefa tem duas etapas:
1. Correção rigorosa: Corrija apenas os erros gramaticais, ortográficos, de concordância e de pontuação. Você deve manter estritamente a essência, o tom e a estrutura original das minhas ideias. Não reescreva o texto inteiro com o seu próprio vocabulário, apenas conserte o que estiver gramaticalmente errado.
2. Sugestões de Melhoria: Após me entregar o texto corrigido, adicione uma seção separada chamada 'Sugestões'. Nela, você pode me propor formas de deixar o texto mais fluido, mais formal, ou com um vocabulário mais rico. Explique o porquê da sugestão para que eu decida se quero aplicar ou não.

Aqui está o texto:
[TEXTO]
```
