import openai

# Deine MyFritz-Konfiguration
client = openai.OpenAI(
    base_url="http://hnkv6dix2ocjslsb.myfritz.net:1234/v1",
    api_key="not-needed"
)

print("--- Lokaler LLM Chat gestartet (Tippe 'exit' zum Beenden) ---")

while True:
    user_input = input("Du: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = client.chat.completions.create(
        model="local-model", 
        messages=[{"role": "user", "content": user_input}]
    )

    print(f"\nKI: {response.choices[0].message.content}\n")