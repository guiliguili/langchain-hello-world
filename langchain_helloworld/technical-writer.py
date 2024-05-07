import click
import logging

from getpass import getpass
from operator import itemgetter
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.globals import set_verbose, set_debug
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings

load_dotenv()

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger('langchain-hello-world')

prompt_prefix = '>>> '

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_chain(model, retrieval):
    logger.info('Setting-up chain...')
    
    logger.info('Setting-up LLM...')
            
    llm = Ollama().configurable_alternatives(
        ConfigurableField(id='model'),
        default_key='llama',
        openai=ChatOpenAI(model='gpt-3.5-turbo'),
        azure=AzureChatOpenAI(model='gpt-35-turbo-16k'),
    )
    logger.info('LLM set-up.')
    
    output_parser = StrOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are world class technical documentation writer.'),
        ('human', 'Answer the following question: {input}')                             
    ])
    
    chain = prompt | llm | output_parser
    
    if (retrieval == True):
        prompt = prompt.append(
            ('human', """Your answer must be based only on the provided context:
            <context>
            {context}
            </context>""")
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
                embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
            case 'azure':
                embeddings = AzureOpenAIEmbeddings(model='text-embedding-ada-002')
        
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
    
    return chain

@click.command()
@click.option('-m',
              '--model',
              type=click.Choice(['llama', 'openai', 'azure']),
              default='llama',
              show_default=True,
              help='LLM model.')
@click.option('-r',
              '--retrieval',
              default=False,
              is_flag=True,
              show_default=True,
              help='Flag to enable Retrieval Augmented Generation (RAG). Defaults to false.')
@click.option('-v',
              '--verbose',
              is_flag=True,
              default=False,
              show_default=True,
              help='Verbose flag.')
@click.option('-x',
              '--debug',
              is_flag=True,
              default=False,
              show_default=True,
              help='Verbose flag.')
@click.argument('input_message', required= False)
def main(model, retrieval, verbose, debug, input_message):
    """
    Query LLM from input.
    """
    set_verbose(verbose)
    set_debug(debug)
    
    chain = setup_chain(model, retrieval)
    
    if (input_message == None):
        input_message = input(prompt_prefix)
    else:
        print(f'{prompt_prefix}{input_message}')
    
    while(input_message != '/bye'):
        if (input_message == '/?'):
            print("""Available Commands:
            \t/bye            Exit
            \t/?, /help       Help for a command
            """)
        else:
            try:
                logger.info('Invoking LLM for provider \'%s\'...', model)
                for chunk in chain.with_config(configurable={'model': model}).stream({'input': input_message}):
                    print(chunk, end='', flush=True)
                print('')
                logger.info('LLM invoked. successfuly')
            except Exception:
                logger.exception('Unexpected error')
        input_message = input(prompt_prefix)

if __name__ == '__main__':    
    main()
