# The list of subjects this app currently has knowledge bases for.
# Add a new subject name here once you've ingested notes for it.
SUBJECTS = [
    "reinforcement_learning",
]

# Defines how many units/topics and total marks each exam type covers,
# matching real VTU-style internals vs end-sem structure.
EXAM_BLUEPRINTS = {
    "internals": {"num_units_covered": 2, "total_marks": 30},
    "end_sem": {"num_units_covered": 5, "total_marks": 100},
}
