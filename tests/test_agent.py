from app.agent.graph import ask_agent

question = "What is the exploration-exploitation tradeoff?"
answer = ask_agent("reinforcement_learning", question)

print("Question:", question)
print()
print("Answer:")
print(answer)
