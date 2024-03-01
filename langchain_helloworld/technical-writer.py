import logging
from getpass import getpass
from operator import itemgetter

import click

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings

from openai import RateLimitError

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@click.command()
@click.option('-i',
              '--input-message',
              type=str,
              required=True,
              help='Input message.')
@click.option('-m',
              '--model',
              type=click.Choice(['llama', 'openai', 'azure']),
              default='llama',
              help='LLM model. Defaults to llama.')
@click.option('-r',
              '--retrieval',
              type=bool,
              default=False,
              help='Flag to enable Retrieval Augmented Generation (RAG). Defaults to false.')
def main(input_message, model, retrieval):
    """
    Query LLM from input.
    """
    logger = logging.getLogger('langchain-hello-world')
    
    logger.info('Setting-up LLM...')
    azure_endpoint = 'dummmy'
    openai_api_version = 'dummmy'
    api_key = 'dummmy'
    match model:
        case 'openai':
            api_key = getpass(prompt='Open AI API key (c.f. https://platform.openai.com/account/api-keys): ')
        case 'azure':
            azure_endpoint = input(  "Azure Open AI endpoint URL : ")
            openai_api_version=input("Azure Open AI API version  : ")
            api_key = getpass(prompt='Azure Open AI API key      : ')
            
    llm = Ollama().configurable_alternatives(
        ConfigurableField(id='model'),
        default_key='llama',
        openai=ChatOpenAI(model='gpt-3.5-turbo', api_key=api_key),
        azure=AzureChatOpenAI(
            model='gpt-35-turbo-16k',
            api_key=api_key,
            azure_endpoint=azure_endpoint, 
            openai_api_version=openai_api_version
        ),
    )
    logger.info('LLM set-up.')
    
    logger.info('Setting-up chain...')
    
    output_parser = StrOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are world class technical documentation writer.')])
    chain = prompt | llm | output_parser
    
    if (retrieval == False):
        prompt.append(('user', 'Answer the following question: {input}'))
    else:
        prompt = prompt.append(
            ('user', """Answer the following question based only on the provided context:
            <context>
            {context}
            </context>

            Question: {input}""")
        )
        
        logger.info('Setting-up retrieval...')
        
        logger.info('Loading documents...')
        loader = WebBaseLoader('https://docs.smith.langchain.com')
        docs = loader.load()
        logger.info('Documents loaded.')
        
        match model:
            case 'llama':
                embeddings = OllamaEmbeddings()
            case 'openai':
                embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=api_key)
            case 'azure':
                embeddings = AzureOpenAIEmbeddings(
                    model='text-embedding-ada-002',
                    api_key=api_key, 
                    azure_endpoint=azure_endpoint, 
                    openai_api_version=openai_api_version
                )
        
        logger.info('Splitting documents...')
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        logger.info('Documents splitted.')
        
        logger.info('Vectorizing documents...')
        vector = FAISS.from_documents(documents, embeddings)
        logger.info('Documents vectorized.')
        
        retriever = vector.as_retriever()
        
        setup_and_retrieval = {
            "context": itemgetter('input') | retriever | format_docs, 
            "input": itemgetter('input')
        }
        chain = setup_and_retrieval | chain
        
        logger.info('Retrieval setup.')
    
    logger.info('Chain set-up.')

    logger.info('Invoking LLM for provider \'%s\'...', model)
        
    try:
        for chunk in chain.with_config(configurable={'model': model}).stream({'input': input_message}):
            print(chunk, end='', flush=True)
        print('')
        logger.info('LLM invoked. successfuly')
    except RateLimitError as e:
        logger.error('Rate limit error: %s', e.message)
    except Exception:
        logger.exception('Unexpected error')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
