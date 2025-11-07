# This is check_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

try:
    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key is None:
        print("--- ERROR ---")
        print("Could not find GEMINI_API_KEY in your .env file.")
        print("Please make sure your .env file is in the correct folder and the key is set.")
    else:
        genai.configure(api_key=api_key)
        
        print("Finding available models for your API key...")
        
        # List all models and find the ones that work for 'generateContent'
        found_model = False
        for model in genai.list_models():
          if 'generateContent' in model.supported_generation_methods:
            print(f"  -> Found usable model: {model.name}")
            found_model = True
            
        if not found_model:
            print("\n--- ERROR ---")
            print("No models were found for your key.")
            print("This usually means your Google Cloud Project is missing a step:")
            print("1. Did you enable the 'Generative Language API'?")
            print("2. Did you link a 'Billing Account' to your project?")
            print("Please double-check those two steps in the Google Cloud Console.")
        else:
            print("\n--- SUCCESS! ---")
            print("Please copy one of the model names from the list above (e.g., 'gemini-1.5-flash')")
            print("and paste it into your 'ai_agent.py' file.")

except Exception as e:
    print(f"\n--- UNEXPECTED ERROR ---")
    print(f"An error occurred: {e}")
    print("This might be a network issue or an invalid API key. Please check your key in the .env file.")