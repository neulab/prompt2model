import openai

def generate_example(prompt: str) -> openai.Completion:
    """Generate a response using OpenAI's gpt-3.5-turbo.

    Args:
        prompt: A prompt asking for a response.

    Returns:
        A response object.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
    )
    return response