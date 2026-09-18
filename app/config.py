# The list of subjects this app currently has knowledge bases for.
SUBJECTS = [
    "reinforcement_learning",
    "autoencoders_and_generative_ai",
]

# Real exam blueprints, matching this college's actual CIE (internals)
# and end-sem paper structure - including compulsory vs choice-based
# questions, not just a mark total.
EXAM_BLUEPRINTS = {
    "internals": {
        "total_marks": 50,
        "structure": [
            {
                "part": "Part A",
                "instructions": "Answer BOTH Q1(a) and Q1(b). Compulsory.",
                "questions": [
                    {"label": "Q1(a)", "marks": 5},
                    {"label": "Q1(b)", "marks": 5},
                ],
            },
            {
                "part": "Part B",
                "instructions": "Answer BOTH Q2(a) and Q2(b) (compulsory), AND choose EITHER Q2(c) OR Q2(d).",
                "questions": [
                    {"label": "Q2(a)", "marks": 5},
                    {"label": "Q2(b)", "marks": 5},
                    {"label": "Q2(c) [choice with Q2(d)]", "marks": 10},
                    {"label": "Q2(d) [choice with Q2(c)]", "marks": 10},
                ],
            },
            {
                "part": "Part C",
                "instructions": "Answer EITHER the full Q3 (both Q3(a) and Q3(b)) OR the full Q4 (both Q4(a) and Q4(b)).",
                "questions": [
                    {"label": "Q3(a) [choice with Q4]", "marks": 10},
                    {"label": "Q3(b) [choice with Q4]", "marks": 10},
                    {"label": "Q4(a) [choice with Q3]", "marks": 10},
                    {"label": "Q4(b) [choice with Q3]", "marks": 10},
                ],
            },
        ],
    },
    "end_sem": {
        "total_marks": 100,
        "structure": [
            {
                "part": f"Unit {i}",
                "instructions": f"Answer EITHER Q{2*i-1} OR Q{2*i}. Both cover Unit {i}.",
                "questions": [
                    {"label": f"Q{2*i-1} [choice, Unit {i}]", "marks": 20},
                    {"label": f"Q{2*i} [choice, Unit {i}]", "marks": 20},
                ],
            }
            for i in range(1, 6)
        ],
    },
}
