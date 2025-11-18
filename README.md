# Projeto Final Processamento Digital de Imagens
Este repositório contém o código e os recursos do projeto final da disciplina de Processamento Digital de Imagens. O objetivo deste projeto é identificar moedas sob diferentes condições de iluminação, superfícies e ângulos de visão, utilizando técnicas avançadas de processamento de imagens, e calcular o valor total das moedas identificadas. 

# Estrutura do Repositório
- `coins/`: Contém imagens de moedas utilizadas como referência para identificação.
- `test_images/`: Contém imagens de teste para validação do algoritmo.
- `CoinCount.py`: Script principal que implementa o algoritmo de identificação e contagem de moedas.
- `README.md`: Documentação do projeto.

# Algoritmo

### 1. Pré-processamento
Em seguida, a imagem é convertida para escala de cinza e suavizada para reduzir ruído, variações de iluminação e reflexos.

## 2. Correção de Perspectiva
A primeira etapa do algoritmo é a correção de perspectiva da imagem para garantir que as moedas estejam alinhadas de forma adequada, facilitando a detecção e identificação.
### 2.1 Detecção de Elipses
### 2.2 Identificação dos Pontos de Interesse
### 2.3 Definição de circulos de referência
### 2.4 Cálculo da Homografia
### 2.5 Aplicação da Transformação de Perspectiva 

### 3. Detecção de Padrões por Bordas
A detecção de bordas é realizada para identificar os contornos e padrões das moedas na imagem.

### 4. Alinhamento de padrões
Os padrões das moedas são alinhados com os padrões de referência armazenados na pasta `coins/` para facilitar a comparação.

### 5. Comparação e Identificação
O algoritmo compara os padrões detectados na imagem com os padrões de referência para identificar o tipo e o valor de cada moeda.

### 6. Cálculo do Valor Total
Por fim, o valor total das moedas identificadas é calculado e exibido.
