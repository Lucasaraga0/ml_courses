## Aula 1 - 10/09

### Perceptron do Rosenblatt

- Introduziu pesos reais e a utilização do algoritmo de aprendizagem:
    - Pesos w e b são números reais ajustáveis
    - O algoritmo de aprendizagem é uma regra para ajustar os pesos baseado nos dados observados

- Função de ativação: Função degrau, para classificação binária

    $$\hat{y} = \sigma(w^Tx + b) = \begin{cases} 1 & \text{se } w^Tx +b \geq 0 \\ 0 & \text{se } w^Tx +b \leq 0 \end{cases}$$

- Regra de aprendizado: o meio como o modelo irá aprender, baseado no erro do valor predito por ele, $\hat{y_i}$ em relação ao valor real, $y_i$. 
    $$w^{(t+1)} \leftarrow w^{(t)} + \eta(y_n - \hat{y_n}) \tilde{x_n}$$
    
- A regra de aprendizado pode ser derivada como descida de gradiente estocástica em uma função de custo, nesse caso o *critério do perceptron*.
    - Esse critério diz que, baseando-se em alvos $y_n \in \{-1,1\}$, um ponto é classificado incorretamente se seu sinal da saída for diferente do sinal do alvo.
    - O custo será o resultado da soma de todos os pontos mal classificados $\mathcal{M}$:
    $$J_P(w) = \sum_{n \in \mathcal{M}} -y_n(w^T x_n)$$

    - Depois do cálculo do gradiente, a regra de atualização da descida de gradiente é:
    $$w^{(t+1)} = w^{(t)} + \eta y_n x_n$$

- O perceptron, no entanto, tinha limitações, sendo o caso mais notável o problema da porta lógica XOR.

### Modelos de camada única

- Possuem a estrutura: 
    - Entradas: normalmente os dados, mas também pode ser um mapeamento.
    - Combinação linear: usa uma soma ponderada das entrada $z=w^T \phi (x) + b$
    - Função de ativação: Função $\sigma$ usada para produzir uma saída do tipo $\hat{y} = \sigma(z)$.
    - Função de custo: $\mathcal{J} (w,b)$, para medir a perda/custo para os parâmetros da rede.

- Regressão x Classificação
- Tradeoff viés e variância