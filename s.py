import google.generativeai as genai
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyATvhsUfFxouR-6HJmiC73RO-rIrQx3BAE"  # Replace with your actual API key

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

try:

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello, world!")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")