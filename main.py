from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
## https://www.thepythoncode.com/article/conversational-ai-chatbot-with-huggingface-transformers-in-python


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # model_name = "microsoft/DialoGPT-large"
    model_name = "microsoft/DialoGPT-medium"
    # model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # chatting 5 times with sampling
    for step in range(5):
        # take user input
        text = input(">> You:")
        # encode the input and add end of string token
        input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
        # concatenate new user input with chat history (if there is)
        bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
        # generate a bot response
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            do_sample=True,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id
        )
        # print the output
        output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"DialoGPT: {output}")