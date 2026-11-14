from llama_cpp import Llama

MODEL_PATH = "D:/AImodels/llama3.1-heretic-unsensored.Q6_K.gguf"

# --- Сколько слоёв реально влезет в VRAM ---
N_GPU_LAYERS = 20  

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=9216,
    n_gpu_layers=N_GPU_LAYERS,   # часть слоёв на GPU, остальное на CPU
    n_threads=8,                 # потоки CPU — важно, т.к. часть слоёв на CPU
    n_batch=256,                 # меньше батч = меньше пиковой памяти
    use_mmap=True,               # модель маппится из файла, не грузится целиком в RAM
    use_mlock=False,             # не блокируем RAM
    offload_kqv=True,            # KQV-кэш на GPU (сильно ускоряет attention)
    chat_format="chatml",
    verbose=False,
)

SYSTEM_PROMPT = (
    """
    Ты - помогающий агент
    отвечай кратко и по смыслу, если тебя не просят иное
    если зациклился на одном слове, то извинись и продолжи
    """
)

print("Агент запущен. Введите 'exit' для выхода.\n")
history = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    print("Agent: ", end="", flush=True)
    full_response = ""
    stream = llm.create_chat_completion(
        messages=history,
        max_tokens=1024,
        temperature=0.6,
        seed=100,
        stream=True,
    )
    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        token = delta.get("content", "")
        if token:
            print(token, end="", flush=True)
            full_response += token
    print("\n")

    history.append({"role": "assistant", "content": full_response})