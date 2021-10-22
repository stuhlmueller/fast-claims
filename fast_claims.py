import asyncio
import os
import streamlit as st
import spacy
import openai
import requests
import serpapi
import simplet5

from sentence_transformers import CrossEncoder
from dataclasses import dataclass
from datetime import datetime

import papers
import prompts


semantic_scholar_api_key = os.environ["semantic_scholar_api_key"]
serpapi_api_key = os.environ["serpapi_api_key"]
openai_api_key = os.environ["openai_api_key"]

openai.api_key = openai_api_key

semantic_scholar_headers = {"x-api-key": semantic_scholar_api_key}
openai_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}",
}


@st.experimental_singleton
def get_msmarco_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512)


@st.experimental_singleton
def get_spacy_nlp():
    return spacy.load("en_core_web_sm")


@st.experimental_singleton
def get_t5_oneline_summary():
    model = simplet5.SimpleT5()
    model.load_model("t5","snrspeaks/t5-one-line-summary")
    return model


msmarco_encoder = get_msmarco_encoder()
nlp = get_spacy_nlp()
t5_oneline_summary = get_t5_oneline_summary()


@st.cache(persist=True, allow_output_mutation=True)
def get_papers(question, n=10):
    params = {
        "engine": "google_scholar",
        "q": question,
        "api_key": serpapi_api_key,
        "num": min(n * 2, 20),
    }
    search = serpapi.GoogleSearch(params)
    data = search.get_dict()
    scholar_results = data["organic_results"]
    all_papers = []
    for scholar_result in scholar_results:
        title = scholar_result.get("title")
        if not title:
            continue
        params = {"query": title}
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=semantic_scholar_headers,
        )
        response_json = response.json()
        datum = response_json.get("data")
        if not datum:
            continue
        paper_id = datum[0].get("paperId")
        params = {"fields": "title,abstract,venue,authors,citationCount,url,year"}
        paper_detail_response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
            params=params,
            headers=semantic_scholar_headers,
        )
        paper_detail = paper_detail_response.json()
        abstract = paper_detail.get("abstract")
        if not abstract:
            continue
        all_papers.append(papers.Paper(title=title, abstract=abstract))
        if len(all_papers) >= n:
            break
    return all_papers


def compress_claim_text_finetuned(question, claim_text, model):
    prompt = prompts.fast_claim_compress_prompt.format(
        question=question, claim_text=claim_text
    )

    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 200,
        "stop": ["<end>", "\n", '"'],
        "temperature": 0,
    }
    response = requests.post(
        f"https://api.openai.com/v1/completions",
        json=data,
        headers=openai_headers,
    )
    completion_result = response.json()
    choices = completion_result.get("choices")
    if not choices:
        return ""
    return choices[0]["text"].strip()


def compress_claim_text_instruct(question, claim_text):
    prompt = prompts.claim_compress_prompt.format(
        question=question, claim_text=claim_text
    )
    engine = "davinci-instruct-beta-v2"
    data = {
        "prompt": prompt,
        "max_tokens": 200,
        "stop": ["<end>", "\n", '"'],
        "temperature": 0,
    }
    response = requests.post(
        f"https://api.openai.com/v1/engines/{engine}/completions",
        json=data,
        headers=openai_headers,
    )
    completion_result = response.json()
    choices = completion_result.get("choices")
    if not choices:
        return ""
    return choices[0]["text"].strip()


def compress_claim_text_t5(question, claim_text):
    return t5_oneline_summary.predict(claim_text)[0]


def compress_claim(question, claim, model, input_type):
    if model == "best sentence":
        return claim.text
    if input_type == "best sentence":
        input_text = claim.text
    elif input_type == "full abstract":
        input_text = claim.paper.abstract
    else:
        raise ValueError(input_type)
    if model == "t5-one-line-summary":
        return compress_claim_text_t5(question, input_text)
    elif model == "davinci-instruct-beta-v2-few-shot":
        return compress_claim_text_instruct(question, input_text)
    else:
        return compress_claim_text_finetuned(question, input_text, model)


@dataclass(order=True)
class Claim:
    text: str
    paper: papers.Paper

    def __repr__(self):
        return self.text

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        if not isinstance(other, Claim):
            return False
        return self.text == other.text


def paper_to_claims(paper):
    claim_texts = [sent.text for sent in nlp(paper.abstract).sents]  # [paper.title] +
    return [Claim(text=text, paper=paper) for text in claim_texts]


def main():

    question = st.text_input("Question", "How can I summarize long documents?")

    num_papers_available = 15
    num_papers_shown = 5

    question_papers = get_papers(question, n=num_papers_available)

    start = datetime.now()

    # 1. Extract the sentences of all {num_papers_available} papers
    all_claims = set()
    for (i, paper) in enumerate(question_papers):
        # with st.expander(paper.title):
        #     st.write(paper.abstract)
        claims = set(paper_to_claims(paper))
        all_claims = all_claims | claims

    # 2. Rank all sentences using babbage based on the question
    all_claims = list(all_claims)
    openai_search_results = openai.Engine("babbage").search(
        documents=[claim.text for claim in all_claims], query=question
    )["data"]
    scores = [result["score"] for result in openai_search_results]
    scored_claims = zip(scores, all_claims)
    sorted_claims = [claim for (score, claim) in sorted(scored_claims, reverse=True)]

    # 3. Create a subset of candidate sentences, starting with
    #    the best babbage sentences, until we cover {num_papers_shown} papers
    seen_papers = set()
    top_babbage_claims = []
    for claim in sorted_claims:
        top_babbage_claims.append(claim)
        seen_papers.add(claim.paper)
        if len(seen_papers) >= num_papers_shown:
            break

    # 4. Rank the subset using msmarco
    top_babbage_texts = [claim.text for claim in top_babbage_claims]
    scores = msmarco_encoder.predict([(question, text) for text in top_babbage_texts])
    scored_claims = zip(scores, top_babbage_claims)
    sorted_claims = [claim for (score, claim) in sorted(scored_claims, reverse=True)]

    # 5. Select summarization model
    summarization_model = st.selectbox(
        "Summarization model",
        options=[
            "best sentence",
            "curie:ft-ought-1-2021-10-22-00-52-45",
            "babbage:ft-ought-1-2021-10-22-01-05-15",
            "ada:ft-ought-1-2021-10-22-00-42-58",
            "t5-one-line-summary",
            "davinci-instruct-beta-v2-few-shot"
        ],
    )
    if summarization_model != "best sentence":
        summarization_input = st.selectbox(
            "Summarization input",
            options=[
                "best sentence",
                "full abstract"
            ]
        )
    else:
        summarization_input = "full abstract"

    # 6. Use the best sentence for each paper
    seen_papers = set()
    for claim in sorted_claims:
        if claim.paper in seen_papers:
            continue
        short_claim = compress_claim(
            question=question, claim=claim, model=summarization_model, input_type=summarization_input
        )
        with st.expander(short_claim):
            st.write(claim)
            st.write(claim.paper.title)
            st.write(claim.paper.abstract)
        seen_papers.add(claim.paper)

    elapsed = datetime.now() - start
    st.write(
        f"Claim extraction: {elapsed.seconds + elapsed.microseconds/1000000:.3f} seconds (not parallelized)"
    )


if __name__ == "__main__":
    main()
