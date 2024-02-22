import logging
from getpass import getpass
import os

import click

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from openai import RateLimitError

@click.command()
@click.option('-i',
              '--input-message',
              type=str,
              required=True,
              help="Input message.")
@click.option('-p',
              '--provider',
              type=click.Choice(['LLAMA', 'OPEN_AI']),
              default='LLAMA',
              help="LLM provider. Defaults to LLAMA.")
def main(input_message, provider):
    """
    Query LLM from input.
    """
    logger = logging.getLogger('langchain-hello-world')
    
    logger.info('Setting-up LLM for provider \'%s\'...', provider)
    match provider:
        case 'LLAMA':
            llm = Ollama(model="llama2")
        case 'OPEN_AI':
            OPENAI_API_KEY = getpass(prompt='Open AI API key (c.f. https://platform.openai.com/account/api-keys): ')
            llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    logger.info('LLM set-up.')
    
    logger.info('Invoking LLM...')
    
    try:
        response = chain.invoke({"input": input_message})
        logger.info('LLM invoked. successfuly')
    
        print(response)
    except RateLimitError as e:
        logger.error('Rate limit error: %s', e.message)
    except Exception:
        logger.exception('Unexpected error')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
