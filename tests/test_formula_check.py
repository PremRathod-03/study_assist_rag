from app.loaders import load_file

pages = load_file('data/raw_notes/reinforcement_learning/unit1_rl.pdf')
for i, page in enumerate(pages):
    if "bellman" in page["text"].lower() or "value function" in page["text"].lower():
        print(f"--- page {i} ---")
        print(page["text"])
        print()
