# 🧠 Experimento Cartesiano Computacional  
**Análise filosófica através de LLMs com método cartesiano**  

![License](https://img.shields.io/badge/License-MIT-green) 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![LLM](https://img.shields.io/badge/LLM-llama.cpp-yellow)

Este projeto implementa uma análise filosófica seguindo o método cartesiano, utilizando modelos de linguagem (LLM) para processar questões complexas. O objetivo é decompor uma pergunta filosófica em subquestões elementares, gerar análises iniciais e refiná-las em vários ciclos para obter uma compreensão mais profunda do problema.

## Estrutura do Projeto

O projeto é composto pelos seguintes arquivos e diretórios:

- **`config_default.json`**: Arquivo de configuração que define parâmetros como o caminho do modelo, número de threads, temperatura, etc.
- **`decartes.py`**: Script principal que implementa a classe `ExperimentoCartesiano` e realiza a análise.
- **`resultados/`**: Diretório onde os resultados das análises são salvos, incluindo logs, análises iniciais e refinamentos.
- **LLM**: DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf versão quantizada

## Como Usar

### 1. Instalação

Para executar o Experimento Cartesiano, você precisará configurar o ambiente corretamente. Siga os passos abaixo:

#### 1.1. Verifique a versão do Python

Certifique-se de ter o Python 3.x instalado. Você pode verificar a versão do Python com o seguinte comando:

```bash
python3 --version
```

Se o Python não estiver instalado, você pode baixá-lo em python.org.

#### 1.2. Crie um ambiente virtual (opcional, mas recomendado)

Para isolar as dependências do projeto, é recomendado criar um ambiente virtual. Execute os seguintes comandos:

```bash
python3 -m venv meu_ambiente  # Cria um ambiente virtual chamado "meu_ambiente"
source meu_ambiente/bin/activate  # Ativa o ambiente virtual (Linux/Mac)
# ou
meu_ambiente\Scripts\activate  # Ativa o ambiente virtual (Windows)
```

#### 1.3. Instale as dependências

O projeto depende de duas bibliotecas principais que trabalham em conjunto:

- **llama-cpp-python**: Uma interface Python para o llama.cpp, permitindo que você execute scripts Python para interagir com modelos de linguagem.
- **llama.cpp**: Uma implementação eficiente em C++ para executar modelos de linguagem, otimizada para desempenho em CPUs e GPUs.

Para garantir que o projeto funcione corretamente e com o máximo de desempenho, você precisará instalar ambos.

##### 1.3.1. Instale o llama-cpp-python

O llama-cpp-python é a biblioteca que permite integrar o llama.cpp com scripts Python. Instale-o via pip:

```bash
pip install llama-cpp-python
```

No entanto, para obter o melhor desempenho, é altamente recomendável compilar o llama.cpp manualmente e configurar o llama-cpp-python para usá-lo.

##### 1.3.2. Compile o llama.cpp manualmente (recomendado)

O llama.cpp é o núcleo que executa os modelos de linguagem de forma eficiente. Compilá-lo manualmente permite otimizações específicas para o seu hardware, como suporte a CUDA para GPUs ou AVX2 para CPUs modernas.

Siga os passos abaixo para compilar o llama.cpp:

1. Clone o repositório do llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

2. Compile o llama.cpp:

   - Se você tiver uma GPU compatível com CUDA, compile com suporte a CUDA para acelerar o processamento:
   
   ```bash
   make LLAMA_CUBLAS=1
   ```

   - Para CPUs modernas com suporte a AVX2, compile com otimizações:
 
   ```bash
   make LLAMA_AVX2=1
   ```

   - Se você não tiver certeza do hardware, compile com suporte básico:
 
   ```bash
   make
   ```

3. Instale o llama-cpp-python com suporte à compilação manual:
   Após compilar o llama.cpp, reinstale o llama-cpp-python apontando para o diretório compilado:

```bash
pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade --no-binary :all:
```

Isso garantirá que o llama-cpp-python use a versão compilada manualmente do llama.cpp, proporcionando o melhor desempenho possível.

##### 1.3.3. Instale o tiktoken

O tiktoken é usado para tokenização avançada e pode ser instalado diretamente via pip:

```bash
pip install tiktoken
```

#### 1.4. Verifique a instalação

Após instalar as dependências, verifique se tudo está funcionando corretamente:

```bash
python3 -c "import llama_cpp; import tiktoken; print('Dependências instaladas com sucesso!')"
```

Se não houver erros, as dependências foram instaladas corretamente.

#### 1.5. Baixe o modelo de linguagem

O projeto utiliza um modelo de linguagem no formato GGUF. Certifique-se de que o caminho do modelo está configurado corretamente no arquivo config_default.json. Por exemplo:

```json
"modelo": {
    "path": "/caminho/para/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
    "n_ctx": 4096,
    "n_threads": 8
}
```

Se você ainda não tem o modelo, pode baixá-lo de fontes confiáveis, como o Hugging Face. Certifique-se de escolher um modelo no formato GGUF, que é otimizado para uso com o llama.cpp.

#### 1.6. Como llama-cpp-python e llama.cpp trabalham juntos

- **llama-cpp-python**: Fornece uma interface Python para interagir com o llama.cpp. Ele permite que você carregue modelos, gere textos e gerencie o processo de análise diretamente em scripts Python.

- **llama.cpp**: É o núcleo que executa os modelos de linguagem de forma eficiente. Ele é responsável por toda a computação pesada, como inferência de modelos e geração de tokens.

Ao compilar o llama.cpp manualmente, você garante que ele esteja otimizado para o seu hardware, enquanto o llama-cpp-python facilita a integração com o código Python do projeto.

#### 1.7. Ganhos de Performance com Compilação Manual

Compilar o llama.cpp manualmente traz vários benefícios:

- **Aceleração em GPUs**: Se você tiver uma GPU compatível com CUDA, a compilação com suporte a CUDA pode acelerar significativamente o processamento.

- **Otimização para CPUs**: Compilar com suporte a AVX2 ou outras instruções modernas de CPU pode melhorar o desempenho em até 2x ou mais.

- **Controle total**: A compilação manual permite ajustar o código para o seu hardware específico, maximizando a eficiência.

Se você não compilar manualmente, o projeto ainda funcionará, mas pode não atingir o desempenho máximo.

### 2. Configuração

Edite o arquivo config_default.json para ajustar os parâmetros do modelo e da análise. 

### 3. Execução

Execute o script com a pergunta filosófica que deseja analisar:

```bash
python decartes.py --pergunta "Podem as máquinas pensar?"
```

Caso deixe em default, a pergunta que estiver dentro do código será executada:

```python
parser.add_argument("--pergunta", type=str, default="Podem as máquinas pensar?")
```

Os resultados serão salvos no diretório resultados/, com arquivos JSON e TXT contendo a análise inicial, refinamentos e logs.

## Exemplo de Saída

### Logs (analise_20250308_010715_log.txt)

```
[2025-03-08 01:07:15] === INÍCIO DA ANÁLISE ===
[2025-03-08 01:07:15] Pergunta: Podem as máquinas pensar?
[2025-03-08 01:07:15] Parâmetros: max_tokens=1024, temperatura=0.75, max_refinamentos=5
...
```

### Análise Final (analise_20250308_010715_final.json)

O arquivo analise_20250308_010715_final.json é o resultado final da análise cartesiana realizada pelo projeto. Ele contém todas as informações geradas durante o processo de análise, incluindo a pergunta original, as subquestões geradas, as respostas iniciais e os refinamentos realizados.

#### Estrutura do Arquivo JSON

O arquivo é organizado da seguinte forma:

```json
{
    "pergunta": "Podem as máquinas pensar?",
    "analise": [
        "Etapa: 1. 0.\nResposta: a. A máquina não tem alma: V  [Lógico]\n             b. A máquina não tem consciência: V\n             ..."
    ],
    "refinamentos": [
        "Considerar as contradições detectadas, pressupostos não questionados e lacunas lógicas na análise original..."
    ],
    "timestamp": "20250308_010715"
}
```

#### Campos Explicados

- **pergunta**:
  - Contém a pergunta filosófica que foi analisada. No exemplo, a pergunta é: "Podem as máquinas pensar?".

- **analise**:
  - É uma lista de respostas geradas para cada subquestão elementar. Cada item na lista representa uma etapa da análise, onde a subquestão é decomposta e respondida.
  - Exemplo:
    ```json
    "analise": [
        "Etapa: 1. 0.\nResposta: a. A máquina não tem alma: V  [Lógico]\n             b. A máquina não tem consciência: V\n             ..."
    ]
    ```
  - Aqui, a subquestão "1. 0." foi respondida com uma análise lógica, indicando que a máquina não tem alma nem consciência.

- **refinamentos**:
  - Contém os ciclos de refinamento aplicados à análise inicial. Cada refinamento é uma tentativa de melhorar a análise, considerando contradições, pressupostos não questionados e lacunas lógicas.
  - Exemplo:
    ```json
    "refinamentos": [
        "Considerar as contradições detectadas, pressupostos não questionados e lacunas lógicas na análise original..."
    ]
    ```
  - Neste caso, o refinamento sugere revisar as contradições e pressupostos da análise inicial.

- **timestamp**:
  - Indica a data e hora em que a análise foi realizada, no formato AAAAMMDD_HHMMSS. No exemplo, o timestamp "20250308_010715" significa que a análise foi realizada em 8 de março de 2025, às 01:07:15.

#### Como o Arquivo é Gerado

O arquivo analise_20250308_010715_final.json é gerado automaticamente pelo script decartes.py após a conclusão da análise. Ele é salvo no diretório resultados/ com um nome único baseado no timestamp, permitindo que várias análises sejam armazenadas sem conflitos.

#### Exemplo Completo

Aqui está um exemplo completo de como o arquivo pode ser estruturado:

```json
{
    "pergunta": "Podem as máquinas pensar?",
    "analise": [
        "Etapa: 1. 0.\nResposta: a. A máquina não tem alma: V  [Lógico]\n             b. A máquina não tem consciência: V\n             ...",
        "Etapa: 2. 1.\nResposta: - A máquina pode executar cálculos complexos, mas não possui intenção ou vontade."
    ],
    "refinamentos": [
        "Refinamento 1: Identificada contradição na definição de 'pensar'. A máquina pode processar informações, mas não possui consciência.",
        "Refinamento 2: Pressuposto não questionado: 'Pensar' requer consciência. Isso pode não ser verdade para todos os tipos de pensamento."
    ],
    "timestamp": "20250308_010715"
}
```

#### Como Usar o Arquivo

- **Revisão da Análise**: Você pode abrir o arquivo JSON em qualquer editor de texto ou visualizador JSON para revisar as respostas e refinamentos.

- **Integração com Outras Ferramentas**: O formato JSON facilita a integração com outras ferramentas ou scripts para análise adicional ou visualização.

- **Comparação entre Análises**: Como cada análise é salva com um timestamp único, você pode comparar diferentes execuções do experimento para ver como a análise evoluiu ao longo do tempo.

## Melhorias Futuras

- Texto final melhor formatado com a resposta à pergunta e um resumo do experimento, seja no começo ou no final.
- Utilização de um modelo mais avançado de LLM (atualmente usa o modelo DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf)
- Otimização do Refinamento: Implementar técnicas para evitar repetições e garantir análises mais concisas.
- Geração de Subquestões: Aprimorar a geração de subquestões para evitar redundâncias.
- Integração com Modelos de Resumo: Utilizar técnicas avançadas de NLP para melhorar o resumo automático.
- Documentação: Expandir a documentação para incluir exemplos detalhados e guias de contribuição.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests com melhorias, correções ou novas funcionalidades.

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
