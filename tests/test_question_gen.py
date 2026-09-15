from app.question_gen import generate_questions

result = generate_questions(
    subject="reinforcement_learning",
    topic="exploration-exploitation tradeoff",
    marks=5,
    count=3,
)
print(result)
