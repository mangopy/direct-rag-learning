from typing import List
import copy

generate_input_prompt = """You are an artificial intelligence assistant. You should gives helpful and precise answers to the user’s questions based on the context. The context here refer to external passages related to user's questions.

Please answer the question: "{question}" using provided passages. Each passage is indicated by a numerical identifier [].
Here are the related passages for your reference.
{docs}
Question: {question}
Your answer: """


user_rank_prompt = """You are RankLLM, an intelligent assistant that can rank passages based on their relevance and usefulness to the user's query.

I will provide you with {n_docs} passages. Please rank these passages based on their usefulness in answering the user's search query: "{question}". 

A passage's usefulness is defined as:
1. Relevance to the question.
2. Contains necessary information to help the user.

The passages are listed below, each with a numerical identifier [].
{docs}

Rank the {n_docs} passages based on their usefulness in descending order. Use the format [] > [], e.g., [2] > [1]. Only respond with the ranking results; do not provide explanations.

Search Query: {question}
Your output: """

def rank_instruction(question: str, docs: List[str]):
    _docs = copy.deepcopy(docs)
    _docs = [f'[{i}] {doc}' for i, doc in enumerate(_docs,1)]
    n_docs = len(_docs)
    _docs = '\n'.join(_docs)
    return user_rank_prompt.format(docs=_docs, question=question, n_docs=n_docs)


def generation_instruction(question: str, docs: List[str]):
    _docs = copy.deepcopy(docs)
    _docs = [f'[{i}] {doc}' for i, doc in enumerate(_docs,1)]
    _docs = '\n'.join(_docs)
    return generate_input_prompt.format(docs=_docs, question=question)


