# Lista 01 - Aprendizado Profundo

- Aluno: Lucas Rodrigues Aragão (538390) - Graduação


## Exercício 4 - Conectando Modelos probabilísticos e Redes Neurais

### A) Regressão Linear Bayesiana

$$\hat{y} = w^Tx + b$$

#### 1. Verossimilhança (Likelihood): Assuma que os dados são gerados com ruído Gaussiano, $y_n ∼ \mathcal{N} (w^T x_n + b, \sigma^2)$ . Escreva a função de log-verossimilhança para um conjunto de dados $\mathcal{D} = \{(x_n, y_n)\}^N _{n=1}.$ Mostre que ela é proporcional ao negativo do Erro da Média dos Quadrados ou Mean Squared Error (MSE) dado por $\mathcal{L} = \frac{1}{N} \sum^N _{n=1} (\hat{y}_n − y_n)$, com $\hat{y}_n = w^T x_n + b$ e $(x_n, y_n) \in \mathcal{D}$ .

A função de verossimilhança é dada por

$$\mathcal{L}(\theta) = \prod_{i=1~}^n P(x_i|\theta)$$

Ao aplicar o log nela tem-se,

$$\log \mathcal{L}(\theta) = \sum_{i=1~}^n \log P(x_i|\theta)$$

Se partimos do principio que $y_i$ são os labels dos nossos dados e $x_i$ o atributo, temos então

$$\log \mathcal{L}(\theta) = \sum_{i=1~}^n \log P(y_i|x_i, \theta) $$

Para $\theta$ sendo igual aos parâmetros do modelo. Se levarmos em consideração que o enunciado diz que $y_n ∼ \mathcal{N} (w^T x_n + b, \sigma^2)$ e que, $\theta = (w, b, \sigma^2)$, temos que 
1. Por estarmos trabalhando com uma distribuição gaussiana 
$$p(y) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\big(-\frac{(y-\mu)^2}{2\sigma^2}\big)}$$
2. Precisamos da média $\mu$, que é dada no enunciado, sendo $\mu = w^TX+b$. Com isso podemos substituir na fórmula 
 $$P(y_n|w,b,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\big(-\frac{(y_n - (w^Tx_n+b))^2}{2\sigma^2}\big)} $$

Ou seja, ao colocarmos isso na nossa função de log verossimilhança teremos que, 

$$\log \mathcal{L}(\theta) = \sum_{i=1~}^n \log{\big[ \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\big(-\frac{(y_n - (w^Tx_n+b))^2}{2\sigma^2}\big)} \big]}$$

Ao aplicar as propriedades logarítmicas temos que,

$$\log \mathcal{L}(w, b, \sigma^2) = \sum_{n=1}^N \big[\frac{1}{2} \log (2\pi \sigma^2) - \frac{(y_n - (w^Tx_n+b))^2}{2\sigma^2} \big]$$

Por fim fazendo algumas simplificações temos,

$$\log{\mathcal{L} (w, b, \sigma^2)} = -\frac{N}{2} \log (2\pi \sigma^2) -\frac{1}{2\sigma^2} \sum^N_{n=1}(y_n - (w^Tx_n + b))^2$$

Observando o MSE, $\mathcal{L} = \frac{1}{N} \sum^N _{n=1} (\hat{y}_n − y_n)^2$, vemos a similaridade do segundo termo da log verossimilhança com o MSE. 

Quando o utilizamos nosso intuito é minimizá-lo, ou seja, fazer que a nossa predição $\hat{y}_n$ seja cada vez mais próxima de $y_n$. Por outro lado, observando a função de log verossimilhança, vemos que nosso objetivo é maximilizá-lo, pelo fato de que, quanto mais próximas nossas predições estiverem, menor será o segundo termo, o que aumentará o valor da função. 

Desse modo, observamos que minimizar o erro médio quadrado é proporcional a maximizar a log verossimilhança;

#### 2. Prior: Assuma uma prior Gaussiana sobre os pesos, $w ∼ \mathcal{N} (0, \alpha^{−1}I)$. Escreva a função de log-prior p(w). Mostre que ela é proporcional ao negativo de um termo de regularização L2 $\lambda\Omega(w) = \lambda||w||_2.$

A priori dada nos mostra que se crê que a média dos pesos do modelo é 0 com variância $\alpha^{-1}$, quanto maior for o nosso $\alpha$ menor deve ser a variância e portanto maior a nossa "crença". Se levamos em conta que $w$ é um vetor de parâmetros, a função densidade de probabilidade deve ser uma gaussiana multivariada, que é dada por

$$p(w) = \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}} \exp(-\frac{1}{2} w^T \Sigma^{-1}w)$$

A matriz de covariância é dada por $\Sigma = \alpha^{-1} I$, e por isso o determinante $|\Sigma|$ é igual a $|\alpha^{-1}I| = \alpha^{-D}$. Além disso a inversa $\Sigma^{-1}$ é $(\alpha^{-1} I)^{-1} = \alpha I$. Substituindo tais valores em na fórmula acima temos, 

$$p(w) = \frac{1}{(2\pi)^{D/2}(\alpha^{-D})^{1/2}} \exp(-\frac{1}{2} w^T\alpha I w) $$

Desenvolvendo as contas, observamos que o $(\alpha^{-D})^{1/2}$ pode ser colocado na parte de cima da fração eliminando o negativo do expoente de $\alpha$. Além disso vemos que dentro do expoente, a conta pode ser simplificada, uma vez que podemos colocar o $\alpha$ no início da multiplificação e tirar a matriz identidade, já que ela é neutra na multiplicação. Com isso obtemos

$$p(w) = \frac{\alpha^{D/2}}{(2\pi)^{D/2}} \exp (-\frac{\alpha}{2} w^T w)$$

Aplicando log obtemos,

$$\log p(w) = \log {\frac{\alpha^{D/2}}{(2\pi)^{D/2}}} -\frac{\alpha}{2} w^T w$$

A segunda parte é o que nos interessa, já que a primeira é uma constante. Se levarmos em conta que $||w||_2 = w^T w$ e supondo um valor de lambda $\lambda = - \frac{\alpha}{2}$ teremos,

$$-\frac{\alpha}{2}w^Tw = \lambda ||w||_2$$

Que é a mesma fórmula do termo de regularização L2. Isso também é intuitivo quando pensamos que o $\alpha$ representa o quanto nós confiamos nas nossas observações, assim como o $\lambda$, que impede o overfitting de modelos. Quanto menor nosso $\alpha$ maior nossa confiança nos dados, assim como no termo de regularização, que quanto menor for o valor do $\lambda$, menor é a interferência na atualização do gradiente. Dessa forma os dois são proporcionais.

#### 3. Posterior (MAP): A estimação do Máximo a Posteriori (MAP) busca os parâmetros que maximizam $p(w|D) ∝ p(D|w)p(w)$. Mostre que maximizar o log-posterior é equivalente a minimizar o Risco Empírico Regularizado da Regressão Ridge (L2) dado por $( w^{\ast} = \arg\min_w \mathcal{L}(w) + \lambda \Omega(w) )$. Que relações temos entre $\alpha$ da priorim, $\sigma^2$, e $\lambda$ da regularização.

Ao fazermos

$$p(w|D) = P(D|w)p(w)$$

Podemos aplicar o logaritmo obtendo

$$\log p(w|D) = \log p(D|w) + \log p(w)$$

Ou seja, a log a posteriori é igual a soma da log verossimilhança com a log priori, que já obtivemos nos itens anteriores. Com isso se substituirmos obtemos, 

$$\log p(w|D) = -\frac{N}{2} \log (2\pi \sigma^2) -\frac{1}{2\sigma^2} \sum^N_{n=1}(y_n - (w^Tx_n + b))^2 + \log {\frac{\alpha^{D/2}}{(2\pi)^{D/2}}} -\frac{\alpha}{2} w^T w $$

Se desconsiderarmos termos constantes obtemos 

$$\log p(w|D) = -\frac{1}{2\sigma^2} \sum^N_{n=1}(y_n - (w^Tx_n + b))^2  -\frac{\alpha}{2} w^T w $$

A "cara" da expressão já é muito similar ao da minimização do risco empírico fornecido. Aqui, entretanto, queremos maximizar essa expressão, porque ,assim como no dito no item 1, isso significa que o negativo do somatório da diferença entre o y previsto e o y real diminuiu. O $\sigma^2$ pode ser assimilado ao valor do passo de aprendizado, usado em algoritmos de descida gradiente, um $\sigma$ muito alto minimiza diminui muito a contribuição do "somatório dos erros", enquanto um sigma muito baixo pode valorizar demais essse somatório. O $\alpha$ da log priori é como o lambda do termo de regularização, que impede que o modelo caminhe demais na direção do gradiente, o que pode causar overfitting.


### B) Regressão de Poisson Bayesiana

#### Verossimilhança e Função de Ligação: Assumimos que os dados seguem uma distribuição de Poisson, $yn ∼ Poisson(\lambda_n)$. Como a taxa $\lambda_n$ deve ser positiva, usamos uma função de ligação exponencial (log-link): $\lambda_n = \exp(w^T x_n)$. Escreva a função de log-verossimilhança.

A distribuição poisson é dada por 

$$p(x=x|\lambda) = \frac{\exp(-\lambda) \lambda^x}{x!}$$

A função de verossimilhança é dada por 

$$\mathcal{L}(\lambda) = \prod_{i=1~}^n P(y_i|\lambda_i)$$

Substituindo temos

$$\mathcal{L}(\lambda) = \prod_{i=1~}^n \frac{\exp(-\lambda) \lambda^{y_i}}{y_i !}$$

Aplicando o log temos

$$\log \mathcal{L}(\lambda) = \sum_{i=1~}^n \log \frac{\exp(-\lambda) \lambda^{y_i}}{y_i !}$$

$$\log \mathcal{L}(\lambda) = \sum_{i=1~}^n \log [\exp(-\lambda) \lambda^{y_i}] - \log y_i !$$

$$\log \mathcal{L}(\lambda) = \sum_{i=1~}^n -\lambda + y_i \log \lambda - \log y_i !$$

Usando a função de ligação fornecida temos 

$$\log \mathcal{L}(\lambda) = \sum_{i=1~}^n -\exp(w^T x_i) + y_i \log [\exp(w^T x_i)] - \log y_i !$$


$$\log \mathcal{L}(\lambda) = \sum_{i=1~}^n -\exp(w^T x_i) + y_i (w^T x_i) - \log y_i !$$


#### 2. Posterior (MAP): Assuma a mesma prior Gaussiana para os pesos, $w ∼ \mathcal{N} (0, \alpha^{−1}I)$. Escreva a expressão para o negativo do log-posterior. Argumente por que este problema não tem uma solução analítica fechada para w.

Como sabemos, 

$$p(w|D) = P(D|w)p(w)$$

$$\log p(w|D) = \log p(D|w) + \log p(w)$$

Utilizando das nossas fórmulas, tanto da log verossimilhança possion quanto da priori gaussiana temos, 


$$\log p(w|D)  = \sum_{i=1~}^n \big[-\exp(w^T x_i) + y_i (w^T x_i) - \log y_i! \big]  + \log {\frac{\alpha^{D/2}}{(2\pi)^{D/2}}} -\frac{\alpha}{2} w^T w$$

Sendo seu negativo,

$$ -\log p(w|D)  =- \sum_{i=1~}^n \big[-\exp(w^T x_i) + y_i (w^T x_i) - \log y_i! \big]  +\frac{\alpha}{2} w^T w$$

Para encontrarmos o w fechado, deveríamos primeiramente encontrar as equações dos gradientes, tanto da verossimilhança quanto da priori. Depois disso a gente somaria essas duas equações e tentaria igualar a zero para poder isolar o w em um lado da equação. Isso não seria possível, porque o w tá ligado a um termo não linear, o $\exp(w^Tx_i)$. Por isso a gente não consegue fazer a solução análitica fechada para encontrar o w.

## Exercício 5: Conceitos

#### 1. O Teorema da Aproximação Universal garante que uma rede neural com uma camada oculta pode aproximar qualquer função. Isso significa que redes profundas (com múltiplas camadas) são redundantes? Explique por que a profundidade é preferível na prática.


Creio que não. É fato que o teorema diz isso, e que com isso, conseguimos aproximar qualquer função. Entretanto, ao adicionarmos mais camadas (por exemplo, camadas ReLu), conseguimos deixar nossa rede neural mais "robusta" e por isso melhorar a nossa aproximação ao utilizarmos transformações não lineares na rede. Portanto, redes de múltiplas camadas não são redundates, pelo fato de conseguirem melhorar a aproximação da função.

#### 2. Explique a conexão entre a Minimização do Risco Empírico Regularizado e o trade-off viés-variância. Como o termo de regularização λΩ(θ) afeta este trade-off? O que podemos dizer sobre o papel da priori e o termo de regularização?

Utilizando uma analogia do professor César, a regularização é como um "freio de mão" que utilizamos quando caminhamos ao gradiente. Com a regularização evitamos enviesar demais nosso modelo aos dados de treinamento ao seguir o algoritmo de descida de gradiente, conseguindo assim, trazer um maior equilíbrio no tradeoff de viés e variância. 


$$\mathbb{S}_{\copyright}$$