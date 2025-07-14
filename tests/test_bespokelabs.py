from datamax.parser.core import DataMax

# 🔐 Replace with your actual credentials
API_KEY = "your-api-key"
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "qwen-turbo"  # or "glm-4", "gpt-4", etc.

# 👇 Test content
test_prompt = "请解释人工智能的基本原理。"
test_content = """
人工智能是计算机科学的一个分支，致力于研究、开发模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。
"""

# ✅ Test call_llm_with_bespokelabs (single prompt)
print("\n=== Testing: call_llm_with_bespokelabs ===")
try:
    result = DataMax.call_llm_with_bespokelabs(
        prompt=test_prompt,
        model_name=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    print("LLM Output:\n", result)
except Exception as e:
    print("Error during LLM call:", e)


# ✅ Test qa_generator_with_bespokelabs (multiple QA chunks)
print("\n=== Testing: qa_generator_with_bespokelabs ===")
try:
    parser = DataMax()
    qa_results = parser.qa_generator_with_bespokelabs(
        content=test_content,
        model_name=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    for i, qa in enumerate(qa_results):
        print(f"\n--- QA Chunk {i + 1} ---\n{qa}")
except Exception as e:
    print("Error during QA generation:", e)
