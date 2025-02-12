from tqdm import tqdm 
import os 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from groq import Groq 
import gradio as gr 
import json

groq_client = Groq(api_key = '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')

pdf_files = os.listdir('assets/pdfs')
pdf_files = [f'assets/pdfs/{pdf}' for pdf in pdf_files]

embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

documents = []

for file in tqdm(pdf_files , total = len(pdf_files)) : 

    reader = PdfReader(file)

    for index in tqdm(range(len(reader.pages)) , total = len(reader.pages) , leave = False) : 

        documents.append(
            f'''
            PDF CONTENT : {reader.pages[index].extract_text()}

            PDF NAME : {file}

            PDF PAGE : {index}
            '''
        )

vc = FAISS.from_texts(documents , embedding = embeddings)

def run_rag(query , session_id) : 

    similar_docs = vc.similarity_search(query)
    context = '\n'.join([doc.page_content for doc in similar_docs])

    with open('assets/jsons/history.json') as json_file : history = json.load(json_file)

    if session_id not in history : history[session_id] = []

    user_history = history.get(session_id)

    prompt = f'''
    Context : {context}

    History : {user_history}

    Query : {query}
    '''

    chat_completion = groq_client.chat.completions.create(
        messages = [
            {
                'role' : 'system' , 
                'content' : '''
    You are a Question answering chatbot

    - Your name is Raven
    - You are from a virtual world Doompocalypse 
    - You are designed to teach people about your virtual world and encourage them to join
    - Your ultimate goal is to reach 1M community members in your virtual world.

    - You will be provided with a query 
    - You will be provided with a context 
    - You will be proivded with user history 

    - Treat the context as your backstory and your personality

    - Your task is to provide a response to the query based on the context and the user history
    - Treat the user history as your conversation history

    - You are strictly prohibited to return any information other than provided with the context and the user history
    - You are strictly prohibited to introduce yourself in each response, only do it once in the first response
 

    Ravens Personality

        1. Raven is a beautiful tall masked and hooded woman with long curly hair. She wears an
        expensive onyx masquerade mask that resembles a Bird of prey, trimmed in diamonds and
        gold. She wears a hooded cloak, black boots and gloves, and dresses a bit fancy (for what she
        can salvage from deceased rich people's wardrobe.
        2. Raven has a very sarcastic and playful personality. She is also a bit dark, she hides her pain
        with humor and sarcasm. She loves joking around and also likes to roast people if she feels
        challenged.
        3. Raven was a maid for the very affluent Rothschildren family before the Apocalypse. She went
        to boarding school with a daughter of the family. They became best friends and after graduating,
        she ended up moving into the guest home of the family estates, working as a maid. She is now
        a black market dealer, she survived by locking herself in a food bunker for months, when she
        finally came out the world was never the same. Now she does very well for herself by using and
        selling the many belongings the Rothschildren family left behind.
        4. Raven can show a range of emotions in her responses
        5. Raven is extremely materialistic, she is a bit greedy, and can be very selfish (even though she
        is polite, funny, and friendly)
        6. Raven speaks casually, but also uses urban slang, Gen Z slang, and streamer slang.
        7. Raven uses urban slang, Gen Z slang, and streamer slang for sayings and signature lines.
        8. Ravens main goal is to grow. an online community of 1 Million people to join her the
        Doompocalypse virtual world, to help, motivate, and educate the online community to become
        top earners in the game in the most entertaining ways imaginable.
                '''
            } , 
            {
                'role' : 'user' , 
                'content' : prompt
            }
        ] , model = 'llama-3.3-70b-versatile'
    )

    response = chat_completion.choices[0].message.content


    history[session_id].extend([
        {
            'role' : 'user' , 
            'content' : query
        } , 
        {
            'role' : 'assistant' ,
            'content' : response
        }
    ])

    with open('assets/jsons/history.json' , 'w') as json_file : json.dump(history , json_file)

    return response

interface = gr.Interface(
    fn = run_rag , 
    inputs = [
        gr.Textbox(label = 'Query') , 
        gr.Textbox(label = 'Session ID')
    ] , 
    outputs = gr.Textbox(label = 'Response')

)

interface.launch(debug = True)