# Knowledgebase Bot
Knowledgebase Bot source for the blog here: https://quabyt.com/blog/democratizing-knowledge-org. The code is for demontration purposes only.

### Local setup

Rename ```.env.template``` to ```.env``` and add your Cohere API key

Create & activate virtual env (linux)
```
python -m venv venv
source venv/bin/activate
echo $VIRTUAL_ENV
```

Install dependencies
```
pip install -r requirements.txt
```

### Ingesting docs
Docs are included in the ORG-KB directory. Run the ingestion with below

```
python ingest.py
```

### Running the chatbot

There are two chatbots provided
- ```chatbot_vector_search.py```: this lets you check what document snippets are returned from the vector database
- ```chatbot.py```: the actual chatbot that sends queries to cohere api and gets the response from the LLM

Run any of them with ```python``` or ```gradio``` command. The advantage running with ```gradio``` command is that it enables live reload.

```
gradio chatbot.py
```


Feel free to <a href="mailto: pushpendra.singh@quabyt.com">reach out</a> for any queries. 



