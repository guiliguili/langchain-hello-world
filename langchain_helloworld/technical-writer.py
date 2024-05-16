import click
import logging
import textwrap
import uuid

from dotenv import load_dotenv

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.globals import set_verbose, set_debug
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings

load_dotenv()

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger('langchain-hello-world')

prompt_prefix = '>>> '

def format_docs(docs):
    """
    Format documents by joining them together separated by 2 lines
    """
    return "\n\n".join(doc.page_content for doc in docs)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Statefully manage chat history
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def setup_chain(model, conversational, retrieval):
    """
    Setup chain
    """
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
        SystemMessage(content='You are a world class technical documentation writer having a conversation with a human.'
            + 'If you do not know the answer to a question, you truthfully say you do not know.')
    ])
    chain = prompt | llm | output_parser
    
    if (retrieval == True):
        logger.info('Setting-up retrieval...')
        
        prompt.append(
            SystemMessagePromptTemplate.from_template(textwrap.dedent("""\
                Your answers are based on the provided context:
                <context>
                {context}
                </context>""")
            )
        )
        
        logger.info('Loading documents...')
        loader = WebBaseLoader('https://docs.smith.langchain.com')
        documents = loader.load()
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
        document_splits = text_splitter.split_documents(documents)
        logger.info('Documents splitted.')
        
        logger.info('Vectorizing documents...')
        vector = FAISS.from_documents(document_splits, embeddings)
        logger.info('Documents vectorized.')
        
        retriever = vector.as_retriever()
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            SystemMessage('Given a chat history and the latest user question which might reference context in the chat history, '
                + 'formulate a standalone question which can be understood without the chat history. '
                + 'Do NOT answer the question, just reformulate it if needed and otherwise return it as is.'
            ),
            MessagesPlaceholder('chat_history'),
            HumanMessagePromptTemplate.from_template('{input}')
        ])
        retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        retrieval = RunnablePassthrough.assign(
            context = retriever | format_docs
        )
        chain = retrieval | chain
        
        logger.info('Retrieval setup.')
        
    if (conversational == True):
        logger.info('Setting-up conversational...')
        
        prompt.append(MessagesPlaceholder('chat_history'))
        
        chain = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="output"
        )
        
        logger.info('Conversational setup.')
    
    prompt.append(
        HumanMessagePromptTemplate.from_template('{input}')
    )
    
    logger.info('Chain set-up.')
    
    return chain

@click.command()
@click.option('-m',
              '--model',
              type=click.Choice(['llama', 'openai', 'azure']),
              default='llama',
              show_default=True,
              help='LLM model.')
@click.option('-c',
              '--conversational',
              default=False,
              is_flag=True,
              show_default=True,
              help='Flag to enable conversational mode.')
@click.option('-r',
              '--retrieval',
              default=False,
              is_flag=True,
              show_default=True,
              help='Flag to enable Retrieval Augmented Generation (RAG).')
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
def main(model, conversational, retrieval, verbose, debug, input_message):
    """
    Query LLM from input.
    """
    set_verbose(verbose)
    set_debug(debug)
    
    session_id = uuid.uuid4()
    
    chain = setup_chain(model, conversational, retrieval)
    chain = chain.with_config(configurable={
        'model': model,
        'session_id': session_id,
        }
    )
    
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
                for chunk in chain.stream({'input': input_message}):
                    print(chunk, end='', flush=True)
                print('')
                logger.info('LLM invoked. successfuly')
            except Exception:
                logger.exception('Unexpected error')
        input_message = input(prompt_prefix)

if __name__ == '__main__':    
    main()
