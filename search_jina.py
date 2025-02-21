"""
This agent is responsible for web search and data retrieval. We wil use jina.ai of this
"""

import requests
from dotenv import load_dotenv
import os
load_dotenv()
jina = os.getenv("JINA_API_KEY")
groq = os.getenv("GROQ_API_KEY")

def readerJina(url):
    url = f'https://r.jina.ai/{url}'
    headers = {'X-Return-Format': 'markdown',
               'X-Token-Budget': '2000'}

    response = requests.get(url, headers=headers)

    return response.text

def searchJina(query):
    query.replace(" ", "%20")
    url = f"https://s.jina.ai/{query}"
    headers = {
    "Authorization": f"Bearer {jina}",
    "X-Token-Budget": "2000",
    "X-Retain-Images": "none"
    }

    response = requests.get(url, headers=headers)
    text = response.text
    # Roughly estimate tokens by splitting on whitespace and punctuation
    tokens = text.split()
    # Take first ~2500 tokens
    truncated_text = ' '.join(tokens[:2500])
    return truncated_text

