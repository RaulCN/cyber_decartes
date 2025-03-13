import json
import time
import os
from llama_cpp import Llama
import tiktoken  # Para tokenização avançada
import requests  # Para chamadas à API do Gemini

class Configuracao:
    """Classe para carregar e gerenciar configurações do arquivo JSON."""
    
    def __init__(self, config_path=None):
        self.dados = {}
        if config_path:
            self.carregar(config_path)
            
    def carregar(self, caminho):
        """Carrega as configurações de um arquivo JSON."""
        try:
            with open(caminho, 'r') as f:
                self.dados = json.load(f)
        except FileNotFoundError:
            raise Exception(f"Arquivo de configuração não encontrado: {caminho}")
            
    def obter(self, *chaves):
        """Obtém um valor das configurações com base nas chaves fornecidas."""
        valor = self.dados
        for chave in chaves:
            valor = valor.get(chave, {})
        return valor
        
    def salvar(self, caminho):
        """Salva as configurações em um arquivo JSON."""
        with open(caminho, 'w') as f:
            json.dump(self.dados, f, indent=4)

class ModeloHandler:
    """Classe para gerenciar interações com o modelo local ou API externa."""
    
    def __init__(self, use_api=False, api_key=None):
        self.use_api = use_api
        self.api_key = api_key
        self.modelo = None

        if not self.use_api:
            # Inicializa o modelo local (mantido no código, mas desativado)
            self.modelo = Llama(
                model_path="/caminho/para/modelo.gguf",  # Mantenha o caminho, mas não será usado
                n_ctx=4096,
                n_threads=8,
                verbose=False
            )

    def gerar_resposta(self, prompt, max_tokens, temperatura):
        if self.use_api:
            # Chama a API do Gemini
            return self._chamar_api_gemini(prompt, max_tokens, temperatura)
        else:
            # Mantém a estrutura do modelo local, mas não executa
            raise Exception("Modelo local desativado. Use a API externa.")

    def _chamar_api_gemini(self, prompt, max_tokens, temperatura):
        """Faz uma chamada à API do Gemini para gerar uma resposta."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperatura
            }
        }

        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            resposta = response.json()
            return resposta["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            raise Exception(f"Erro na API: {response.status_code} - {response.text}")

class ExperimentoCartesiano:
    """Implementa uma análise filosófica seguindo o método cartesiano."""
    
    def __init__(self, config_path=None, use_api=False, api_key=None):
        # Carregar configuração
        self.config = Configuracao(config_path)
        
        # Inicializar o ModeloHandler
        self.modelo_handler = ModeloHandler(
            use_api=use_api,
            api_key=api_key
        )
        
        # Carregar princípios cartesianos
        self.principios = self.config.obter("principios")
        
        # Preparar diretório de saída
        saida_config = self.config.obter("saida")
        self.diretorio_saida = saida_config["diretorio"]
        if not os.path.exists(self.diretorio_saida):
            os.makedirs(self.diretorio_saida)
        
        self.salvar_incremental = saida_config["salvar_incremental"]
        self.intervalo_salvamento = saida_config["intervalo_salvamento"]
        
        # Inicializar logs
        self.logs = []

        # Inicializar tokenizador
        self.tokenizador = tiktoken.get_encoding("cl100k_base")  # Usado para contagem precisa de tokens

    def analisar(self, pergunta):
        """
        Realiza análise cartesiana completa de uma pergunta.
        
        Args:
            pergunta (str): A questão filosófica a ser analisada.
        """
        try:
            # Obter parâmetros de análise
            analise_config = self.config.obter("analise")
            max_tokens = analise_config["max_tokens"]
            temperatura = analise_config["temperatura"]
            max_refinamentos = analise_config["max_refinamentos"]
            
            # Iniciar logs
            self._log("=== INÍCIO DA ANÁLISE ===")
            self._log(f"Pergunta: {pergunta}")
            self._log(f"Parâmetros: max_tokens={max_tokens}, temperatura={temperatura}, max_refinamentos={max_refinamentos}")
            
            # Preparar estrutura de resultados
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            arquivo_base = f"{self.diretorio_saida}/analise_{timestamp}"
            
            resultado = {
                'pergunta': pergunta,
                'analise': [],
                'refinamentos': [],
                'timestamp': timestamp
            }

            # Primeira análise
            self._log("\n[1/3] Gerando análise inicial...")
            analise_inicial = self._processar_etapas(pergunta, max_tokens, temperatura)
            resultado['analise'] = analise_inicial
            
            # Salvar análise inicial
            if self.salvar_incremental:
                self._salvar_resultado(resultado, f"{arquivo_base}_inicial.json")
            
            # Ciclos de refinamento
            self._log(f"\n[2/3] Iniciando {max_refinamentos} ciclos de refinamento...")
            analise_atual = analise_inicial
            
            for ciclo in range(1, max_refinamentos+1):
                self._log(f"\n--- Ciclo {ciclo} de refinamento ---")
                novo_refinamento = self._ciclo_refinamento(analise_atual)
                resultado['refinamentos'].append(novo_refinamento)
                analise_atual = novo_refinamento
                self._log("Refinamento concluído.")
                
                # Salvamento incremental
                if self.salvar_incremental and ciclo % self.intervalo_salvamento == 0:
                    self._salvar_resultado(resultado, f"{arquivo_base}_parcial_{ciclo}.json")
                    self._log(f"Salvamento incremental: ciclo {ciclo}")

            # Salvamento final
            self._log("\n[3/3] Salvando resultados...")
            self._salvar_resultado(resultado, f"{arquivo_base}_final.json")
            
            # Salvar versão em texto para leitura fácil
            with open(f"{arquivo_base}_final.txt", "w") as f:
                f.write(f"Pergunta: {pergunta}\n\n")
                f.write("=== Análise Inicial ===\n")
                f.write("\n".join(resultado['analise']))
                f.write("\n\n=== Refinamentos ===\n")
                for idx, ref in enumerate(resultado['refinamentos'], 1):
                    f.write(f"\nRefinamento {idx}:\n{ref}")
            
            self._log("Análise concluída com sucesso!")
            self._salvar_logs(f"{arquivo_base}_log.txt")

        except Exception as e:
            self._log(f"ERRO FATAL: {str(e)}")
            self._salvar_logs(f"{self.diretorio_saida}/erro_{time.strftime('%Y%m%d_%H%M%S')}_log.txt")
            raise

    def _processar_etapas(self, pergunta, max_tokens, temperatura):
        """
        Processa a pergunta através das etapas do método cartesiano.
        
        Args:
            pergunta (str): A questão a analisar
            max_tokens (int): Número máximo de tokens para cada resposta
            temperatura (float): Valor de temperatura para geração
            
        Returns:
            list: Lista de respostas para cada subquestão
        """
        self._log("\n-> Gerando subquestões elementares...")
        subquestoes = self._gerar_subquestoes(pergunta)
        self._log(f"Subquestões geradas: {len(subquestoes)} itens")
        
        respostas = []
        for idx, etapa in enumerate(subquestoes, 1):
            self._log(f"\n--> Processando etapa {idx}: {etapa[:50]}...")
            prompt = f"""Análise cartesiana:
            Q: {pergunta}
            Foco: {etapa}
            Aplicar {self.principios['duvidar']}:
            """
            resposta = self.modelo_handler.gerar_resposta(prompt, max_tokens, temperatura)
            
            respostas.append(f"Etapa: {etapa}\nResposta: {resposta}")
            self._log(f"Resposta gerada: {resposta[:100]}...")
        
        return respostas

    def _ciclo_refinamento(self, analise_atual):
        """
        Realiza um ciclo de refinamento da análise atual.
        
        Args:
            analise_atual: A análise a ser refinada
            
        Returns:
            str: A análise refinada
        """
        # Juntar a análise atual em uma única string
        analise_completa = " ".join(analise_atual)
        
        # Limitar o número de tokens para não exceder o contexto do modelo
        max_tokens_contexto = self.config.obter("modelo", "n_ctx")  # Limite de tokens do modelo
        max_tokens_analise = max_tokens_contexto - 500  # Reserva 500 tokens para o prompt e refinamento
        
        # Resumir automaticamente a análise para caber no limite de tokens
        analise_resumida = self._resumir_texto(analise_completa, max_tokens_analise)
        
        # Montar o prompt para refinamento
        prompt = f"""Refine a análise considerando:
        1. Contradições detectadas
        2. Pressupostos não questionados
        3. Lacunas lógicas
        
        Análise atual:
        {analise_resumida}
        
        Refinamento:"""
        
        # Gerar o refinamento
        return self.modelo_handler.gerar_resposta(prompt, self.config.obter("analise", "max_tokens"), 0.5)

    def _resumir_texto(self, texto, max_tokens):
        """
        Resumir o texto automaticamente para caber no limite de tokens.
        
        Args:
            texto (str): O texto a ser resumido
            max_tokens (int): Número máximo de tokens permitidos
            
        Returns:
            str: O texto resumido
        """
        tokens = self.tokenizador.encode(texto)
        if len(tokens) <= max_tokens:
            return texto
        
        # Resumir o texto mantendo as partes mais importantes
        tokens_resumidos = tokens[:max_tokens]
        texto_resumido = self.tokenizador.decode(tokens_resumidos)
        return texto_resumido + " [...]"  # Indica que o texto foi resumido

    def _gerar_subquestoes(self, pergunta):
        """
        Decompõe a pergunta principal em subquestões elementares.
        
        Args:
            pergunta (str): A questão principal
            
        Returns:
            list: Lista de subquestões
        """
        prompt = f"""Decomponha '{pergunta}' em subquestões elementares
        seguindo {self.principios['dividir']}:
        1."""
        
        resposta = self.modelo_handler.gerar_resposta(prompt, 300, 0.8)
        
        # Processa resposta do modelo
        return [f"1. {resposta.split('1.')[1]}"] + resposta.split('\n')[1:]

    def _salvar_resultado(self, resultado, caminho):
        """Salva o resultado em formato JSON."""
        with open(caminho, 'w') as f:
            json.dump(resultado, f, indent=4)

    def _log(self, mensagem):
        """Registra mensagens com timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        msg_completa = f"[{timestamp}] {mensagem}"
        print(msg_completa)
        self.logs.append(msg_completa)

    def _salvar_logs(self, caminho):
        """Salva os logs em um arquivo de texto."""
        with open(caminho, "w") as f:
            f.write("\n".join(self.logs))
        self._log(f"Logs salvos em {caminho}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experimento Cartesiano - Análise filosófica via LLM")
    parser.add_argument("--config", type=str, default=None, help="Caminho para arquivo de configuração")
    parser.add_argument("--pergunta", type=str, default="Podem as máquinas pensar?", 
                       help="Pergunta filosófica a ser analisada")
    parser.add_argument("--use_api", action="store_true", help="Usar API externa em vez do modelo local")
    parser.add_argument("--api_key", type=str, default=None, help="Chave da API externa")
    
    args = parser.parse_args()
    
    try:
        # Criar arquivo de configuração padrão se não existir nenhum
        if args.config is None:
            config_default = "config_default.json"
            if not os.path.exists(config_default):
                cfg = Configuracao()
                cfg.salvar(config_default)
                print(f"Criado arquivo de configuração padrão: {config_default}")
            args.config = config_default
                
        # Iniciar experimento
        experimento = ExperimentoCartesiano(
            args.config,
            use_api=args.use_api,
            api_key=args.api_key
        )
        experimento.analisar(args.pergunta)

        print("\n=== EXECUÇÃO CONCLUÍDA ===")
        print(f"Resultados em: {experimento.diretorio_saida}/")
    except Exception as e:
        print(f"ERRO CRÍTICO: {str(e)}")
