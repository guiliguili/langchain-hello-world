# LangChain Hello World

This is a sample project demonstrating the use of [LangChain](https://python.langchain.com/).

The goal is also to apply most of the best practises to apply in [Python](https://www.python.org) projects.

## Installation

1. Install **Conda** using the [miniconda installer](https://conda.io/en/latest/miniconda.html) or [Homebrew](https://brew.sh/index_fr) if you are on macOS
   ```console
   brew install miniconda
   ```
1. Create or update the **langchain-hello-world** conda environment
   ```console
   conda env create -f environment.yml
   [...]
   conda env update -f environment.yml
   ```
1. Activate the **langchain-hello-world** conda environment
   ```console
   conda activate langchain-hello-world
   ```
1. Install [OpenAI integration](https://api.python.langchain.com/en/latest/openai_api_reference.html) for LangChain
   ```console
   pip install langchain-openai
   ```
 
## Running from command line

```console
python -m langchain_helloworld --help
```

## Running Jupyter notebooks

```console
jupyter notebook
```

## Reference

* [LangChain](https://python.langchain.com/)
* [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)
