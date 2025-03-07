# 🧠 Experimento Cartesiano Computacional  
**Análise filosófica através de LLMs com método cartesiano**  

![License](https://img.shields.io/badge/License-MIT-green) 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![LLM](https://img.shields.io/badge/LLM-llama.cpp-yellow)

## 📜 Princípios Chave
```python
# Método Cartesiano Modernizado
1. Dúvida Sistemática: "Rejeitar todas as ideias não claramente verdadeiras"
2. Decomposição: "Dividir cada dificuldade em partes menores"
3. Ordenação: "Conduzir os pensamentos por ordem"
4. Revisão: "Fazer enumerações completas"

##🚀 Instalação

# 1. Clonar repositório
git clone https://github.com/seu-usuario/experimento-cartesiano.git
cd experimento-cartesiano

# 2. Instalar dependências
pip install llama-cpp-python==0.2.23

# 3. Baixar modelo (exemplo)
wget -P ~/modelos https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill/resolve/main/DeepSeek-R1-Distill-Q4_K_M.gguf, ou também baixar via LLMSTUDIO.

##⚙️ Configuração

// config_default.json
{
    "modelo": {
        "path": "~/modelos/DeepSeek-R1-Distill-Q4_K_M.gguf",
        "n_ctx": 2048,
        "n_threads": 2
    },
    "analise": {
        "max_tokens": 768,
        "temperatura": 0.7,
        "max_refinamentos": 5  // 5 ciclos garantidos
    }
}

##🧠 Funcionalidades Principais
- **Decomposição Automática** de questões complexas
- **5 Ciclos de Refinamento** obrigatórios
- **Gestão de Memória Ativa** com `gc.collect()`
- **Saída Unificada** em formato texto/Markdown

##🚀 Execução

=== ANÁLISE FINAL (5 Refinamentos) ===
[Refinamento 1] Identificação de pressupostos ocultos
[Refinamento 2] Eliminação de contradições lógicas
[Refinamento 3] Análise de implicações éticas
[Refinamento 4] Comparação com frameworks morais
[Refinamento 5] Síntese conclusiva validada

Saída salva em txt.

python decartes_7.py \
  --pergunta "Uma máquina pode pensar?" \
  --config config_default.json

##📄 Exemplo de Saída

##📜 Licença MIT

Copyright 2024 Raul Campos Nascimento

Permissão é concedida, gratuitamente, a qualquer pessoa que obtenha uma cópia
deste software e arquivos de documentação associados, para usar, copiar, modificar,
mesclar, publicar, distribuir, sublicenciar e/ou vender cópias do Software.

