import logging
from getpass import getpass
import os

import click

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI

from openai import RateLimitError

@click.command()
@click.option('-i',
              '--input-message',
              type=str,
              required=True,
              help='Input message.')
@click.option('-p',
              '--provider',
              type=click.Choice(['llama', 'openai', 'gpt4']),
              default='llama',
              help='LLM provider. Defaults to llama.')
@click.option('-r',
              '--retrieval',
              type=bool,
              default=False,
              help='Flag to enable Retrieval Augmented Generation (RAG). Defaults to false.')
def main(input_message, provider, retrieval):
    """
    Query LLM from input.
    """
    logger = logging.getLogger('langchain-hello-world')
    
    logger.info('Setting-up LLM...')
    match provider:
        case 'gpt4':
            OPENAI_API_KEY = getpass(prompt='Open AI API key (c.f. https://platform.openai.com/account/api-keys): ')
            os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        case 'openai':
            OPENAI_API_KEY = getpass(prompt='Open AI API key (c.f. https://platform.openai.com/account/api-keys): ')
            os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        case _:
            os.environ['OPENAI_API_KEY'] = 'dummy-key'
            
    llm = Ollama().configurable_alternatives(
        ConfigurableField(id='llm'),
        default_key='llama',
        openai=ChatOpenAI(),
        gpt4=ChatOpenAI(model='gpt-4'),
    )
    logger.info('LLM set-up.')
    
    logger.info('Setting-up chain...')
    if (retrieval == False):
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are world class technical documentation writer.'),
            ('user', '{input}')
        ])
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
    else:
        
        logger.info('Setting-up retrieval chain...')
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}""")
        
        loader = WebBaseLoader('https://docs.smith.langchain.com')
        docs = loader.load()
        
        embeddings = OllamaEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        retriever = vector.as_retriever()
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, document_chain)
        logger.info('Retrieval chain setup.')
    
    logger.info('Chain set-up.')

    logger.info('Invoking LLM for provider \'%s\'...', provider)
        
    try:
        response = chain.with_config(configurable={'llm': provider}).invoke({'input': input_message})
        logger.info('LLM invoked. successfuly')
    
        if (retrieval == False):
            print(response)
        else:
            print(response['answer'])
    except RateLimitError as e:
        logger.error('Rate limit error: %s', e.message)
    except Exception:
        logger.exception('Unexpected error')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
