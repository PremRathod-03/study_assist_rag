from app.agent.graph import ask_agent

question = "What is a value function, and how does it differ from a reward?"
answer = ask_agent("reinforcement_learning", question)

print("Question:", question)
print()
print("Answer:")
print(answer)
