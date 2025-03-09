from llama_cpp import Llama
import os
import re
from datetime import datetime

class AnalisadorSimples:
    def __init__(self, model_path):
        self.modelo = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
        self.diretorio_saidas = "saidas"
        self._criar_diretorio_se_nao_existir()

    def _criar_diretorio_se_nao_existir(self):
        """Cria o diretório de saída se não existir"""
        if not os.path.exists(self.diretorio_saidas):
            os.makedirs(self.diretorio_saidas)

    def _sanitizar_nome_arquivo(self, texto):
        """Remove caracteres inválidos e formata o nome do arquivo"""
        texto = re.sub(r'[\\/*?:"<>|]', '', texto)  # Remove caracteres inválidos
        texto = re.sub(r'\s+', '_', texto)          # Substitui espaços por underscores
        texto = texto.strip()[:50]                  # Limita o comprimento do nome
        return texto or "sem_titulo"                # Garante um nome padrão

    def _gerar_nome_arquivo(self, pergunta):
        """Gera um nome de arquivo único com data/hora"""
        nome_base = self._sanitizar_nome_arquivo(pergunta)
        data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{nome_base}_{data_hora}.txt"

    def _salvar_resposta(self, pergunta, resposta):
        """Salva a pergunta e resposta em um arquivo de texto"""
        nome_arquivo = self._gerar_nome_arquivo(pergunta)
        caminho_completo = os.path.join(self.diretorio_saidas, nome_arquivo)
        
        conteudo = f"Pergunta: {pergunta}\n\nResposta: {resposta}"
        
        with open(caminho_completo, 'w', encoding='utf-8') as arquivo:
            arquivo.write(conteudo)
        
        return caminho_completo

    def analisar(self, pergunta):
        prompt = f"{pergunta}"
        resposta = self.modelo(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )['choices'][0]['text'].strip()
        
        # Salvar em arquivo
        caminho_salvo = self._salvar_resposta(pergunta, resposta)
        
        # Exibir resultados
        print(f"\nPergunta: {pergunta}")
        print(f"\nResposta: {resposta}")
        print(f"\nResposta salva em: {caminho_salvo}")

if __name__ == "__main__":
    modelo_path = "/home/rauto/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"  # Substitua pelo caminho real do seu modelo
    analisador = AnalisadorSimples(modelo_path)
    analisador.analisar("Podem as máquinas pensar?")
