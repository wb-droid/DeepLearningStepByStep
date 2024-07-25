LLM can be conditioned to take up pre-defined characteristics to role-play when chatting. This can be entertaining and also has many practical use cases.

Building on top of TelegramBot, the following is performed to add AI characters for intersting chatting applications.

1. Llama-2 is a powerful family of LLMs. To save resource, 4-bit quantized llama-2-7b-chat.Q4_K_M.gguf is used. Start model and API endpoint.
start_wsl.bat --api --model llama-2-7b-chat.Q4_K_M.gguf

2. AI character can be created by fine-tuning, RAG, or prompt engineering. Here prompt enginerring is used to create a new AI chacter.

3. Llama-2 insert annoying actions during role-playing. Modify the chat bot to remove those actions. Start mirai-chatgpt-bot:
sudo docker run --name mirai-chatgpt-bot -v ./config.cfg:/app/config.cfg -v ./presets/will.txt:/app/presets/will.txt -v ./adapter/chatgpt/api.py:/app/adapter/chatgpt/api.py --network host lss233/chatgpt-mirai-qq-bot:browser-version