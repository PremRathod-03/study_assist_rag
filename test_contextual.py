from app.contextual_chunks import add_context

document = """
Reinforcement learning is a type of machine learning where an agent
learns by interacting with an environment. The agent takes actions,
receives rewards, and updates its behavior over time.
"""

chunk = "The agent's sole objective is to maximize the total reward it receives over the long run."

result = add_context(document, chunk)
print(result)
