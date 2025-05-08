# -*- coding: utf-8 -*-
"""
@author: Debarghya Das

"""

import re
import json
import logging
from typing import List, Dict
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import speech_recognition as sr
import pyttsx3
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Dave")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load NLP model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class Dave:
    def __init__(self, model_name: str = "distilgpt2", max_context: int = 5):
        self.model_name = model_name
        self.max_context = max_context
        self.conversation_history: List[Dict] = []
        self.stop_words = set(stopwords.words('english'))

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
            logger.info(f"Model {model_name} loaded.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def preprocess_input(self, user_input: str) -> str:
        processed = re.sub(r'[^\w\s]', '', user_input.lower().strip())
        tokens = word_tokenize(processed)
        filtered = [token for token in tokens if token not in self.stop_words]
        return ' '.join(filtered)

    def extract_intent(self, user_input: str) -> str:
        keywords = ["code", "program", "script", "function", "algorithm"]
        if any(word in user_input.lower() for word in keywords):
            return "code_request"
        return "general"

    def detect_language(self, user_input: str) -> str:
        if "javascript" in user_input.lower():
            return "javascript"
        elif "c++" in user_input.lower():
            return "cpp"
        elif "java" in user_input.lower():
            return "java"
        else:
            return "python"

    def highlight_code(self, code: str, language: str) -> str:
        try:
            lexer = get_lexer_by_name(language, stripall=True)
            formatter = TerminalFormatter()
            return highlight(code, lexer, formatter)
        except Exception as e:
            logger.error(f"Highlighting error: {e}")
            return code

    def generate_code_response(self, task_description: str) -> str:
        language = self.detect_language(task_description)
        prompt = f"""
Write a {language} program to perform the following task:
{task_description}
Provide only the code, fully working, with necessary functions.
"""
        try:
            response = self.generator(prompt, max_length=400, temperature=0.3, top_p=0.95)[0]['generated_text']
            return self.highlight_code(response.strip(), language)
        except Exception as e:
            logger.error(f"Code generation error: {e}")
            return "Sorry, I couldn't generate the code."

    def generate_response(self, user_input: str, max_length: int = 150) -> str:
        intent = self.extract_intent(user_input)
        if intent == "code_request":
            response = self.generate_code_response(user_input)
        else:
            try:
                processed_input = self.preprocess_input(user_input)
                context = "".join(
                    [f"User: {i['user_input']}\nBot: {i['response']}\n" for i in self.conversation_history[-self.max_context:]]
                ) + f"User: {processed_input}\nBot: "

                generated = self.generator(
                    context,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )[0]['generated_text']

                response = generated.split("Bot: ")[-1].strip()
            except Exception as e:
                logger.error(f"Response generation error: {e}")
                response = "Sorry, I ran into an issue."

        self.update_context(user_input, response)
        return response

    def save_conversation(self, filename: str = "conversation_history.json") -> None:
        try:
            with open(filename, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            logger.info(f"Saved conversation to {filename}")
        except Exception as e:
            logger.error(f"Save failed: {e}")

    def load_conversation(self, filename: str = "conversation_history.json") -> None:
        try:
            with open(filename, 'r') as f:
                self.conversation_history = json.load(f)
            logger.info(f"Loaded conversation from {filename}")
        except Exception as e:
            logger.error(f"Load failed: {e}")

    def summarize_conversation(self) -> str:
        return "\n".join(
            [f"[{msg['timestamp']}] You: {msg['user_input']} -> Dave: {msg['response']}" for msg in self.conversation_history]
        )

    def get_intents_summary(self) -> Dict[str, int]:
        intent_summary = {}
        for msg in self.conversation_history:
            intent = msg.get("intent", "unknown")
            intent_summary[intent] = intent_summary.get(intent, 0) + 1
        return intent_summary

def listen_to_user() -> str:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return ""

def speak_text(text: str) -> None:
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    dave = Dave()
    print("Hi, I'm Dave — your coding assistant! Ask me to write code in Python, C++, JavaScript, etc.")
    print("Commands: 'quit', 'save', 'summary', 'intents', or switch to voice by typing 'voice'.")

    while True:
        mode = input("Type 'voice' for speech input or press enter to type: ").lower()

        if mode == 'voice':
            user_input = listen_to_user()
            if not user_input:
                print("Couldn't hear anything.")
                continue
            print(f"You (spoken): {user_input}")
        else:
            user_input = input("You: ")

        if user_input.lower() == 'quit':
            print("Bye! Talk to you later.")
            break
        elif user_input.lower() == 'save':
            dave.save_conversation()
            print("Conversation saved.")
            continue
        elif user_input.lower() == 'summary':
            print(dave.summarize_conversation())
            continue
        elif user_input.lower() == 'intents':
            print(json.dumps(dave.get_intents_summary(), indent=2))
            continue

        response = dave.generate_response(user_input)
        print(f"Dave: {response}")
        speak_text(response)

if __name__ == "__main__":
    main()
