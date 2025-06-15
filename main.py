from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

model = PeftModel.from_pretrained(base_model, "./model")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = """### Instruction:
Based on the data:
- Name: Jean Pierre
- Email: jean.pierre@lafrance.com
- Probable country: France
Generate probable passwords:

### Response:"""

output = generator(prompt, max_new_tokens=30, do_sample=True, temperature=0.7)
print(output[0]['generated_text'])
