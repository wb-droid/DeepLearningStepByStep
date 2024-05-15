/**
 * Welcome to Cloudflare Workers! This is your first worker.
 *
 * - Run "npm run dev" in your terminal to start a development server
 * - Open a browser tab at http://localhost:8787/ to see your worker in action
 * - Run "npm run deploy" to publish your worker
 *
 * Learn more at https://developers.cloudflare.com/workers/
 */

// Reference: https://blog.devops.dev/free-hosting-for-your-telegram-bot-its-easier-than-you-think-66a5e5c000bb

export default {
    async fetch(request, env, ctx) {
      if(request.method === "POST"){
        const payload = await request.json();
        if('message' in payload){
          const chatId = payload.message.chat.id;
          const input = String(payload.message.text);
          const user_firstname = String(payload.message.from.first_name);
          const system_prompt = "<s>[INST] <<SYS>>You are a helpful AI agent. You answer questions concisely.<</SYS>>"
          const end_prompt = "[/INST]"
          const inputs = system_prompt + input + end_prompt
          var str = await this.query(env.HUGGINGFACE_TOKEN, {"inputs": inputs});
          str = str.replace(inputs, "").split("\n\n")[0];
          const response = str
  
          await this.sendMessage(env.API_KEY, chatId, response);
        }      
      }
      //const response = await this.query(env.HUGGINGFACE_TOKEN, {"inputs": "<s>[INST] <<SYS>>You are a helpful AI agent.<</SYS>>Hi![/INST]"})
      //return new Response('OK' + response);    
      return new Response('OK');    
    },
  
    async query(huggingface_token, data) {
      //const API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
      const API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
      const response = await fetch(
        API_URL,
        {
          headers: { Authorization: "Bearer " + huggingface_token,  'Content-Type': "application/json"},
          method: "POST",
          body: JSON.stringify(data),
        }
      );
      const result = await response.json();
      console.log(result[0].generated_text);
      return result[0].generated_text;
    },
    
    
  
    async sendMessage(apiKey, chatId, text){
      const url = `https://api.telegram.org/bot${apiKey}/sendMessage?chat_id=${chatId}&text=${text}`;
      const data = await fetch(url).then(resp => resp.json());
    }  
  };