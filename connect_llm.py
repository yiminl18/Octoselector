import openai
from google import genai
import anthropic
import config

def GPT_prompt_api(message_content, model_name="gpt-4o-mini"): #gpt-3.5-turbo-0125
    openai.api_key = config.OPENAI_API_KEY 
    response = openai.ChatCompletion.create(
        model = model_name,
        messages = [
            {"role": "user", "content": message_content}],
        temperature = 0.7, ## default 
        max_tokens=4096
    )
    return response["choices"][0]["message"]["content"]

def Gemini_api(prompt, model_name = "gemini-1.5-flash-8b"):
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=model_name, contents=prompt
    )
    return response.text

def Claude_api(prompt,model_name = "claude-3-5-haiku-20241022"):
    client = anthropic.Anthropic(
        api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model= model_name,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

if __name__ == "__main__":
    connect_test_msg = "what is the capital of Tailand"
    ans = GPT_prompt_api(connect_test_msg)
    print("gpt ans: ",ans)
