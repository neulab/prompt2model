import prompt2model.utils.api_tools as api_tools
import os
print(os.getenv("OPENAI_API_KEY"))

api_tools.APIAgent("gpt")