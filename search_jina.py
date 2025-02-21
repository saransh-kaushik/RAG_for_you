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
    headers = {
    "Authorization": "Bearer " + jina,
    }
    
    response = requests.get(url, headers=headers)

    return response.text

def searchJina(query):
    query.replace(" ", "%20")
    url = f"https://s.jina.ai/{query}"
    headers = {
    "Authorization": f"Bearer {jina}",
    "X-Engine": "direct",
    "X-Retain-Images": "none"
    }

    response = requests.get(url, headers=headers)
    return response.text

