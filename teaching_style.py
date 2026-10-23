
TEACHING_STYLE_FEATURES = {
    "communication_explanation": {
        "weight": 0.12,
        "sub_features": [
            "clarity_of_explanation",
            "simple_language",
            "appropriate_speaking_pace",
            "step_by_step_explanation",
        ],
    },
    "student_engagement": {
        "weight": 0.12,
        "sub_features": [
            "asking_questions",
            "encouraging_participation",
            "interactive_teaching",
            "maintaining_attention",
        ],
    },
    "inclusiveness": {
        "weight": 0.10,
        "sub_features": [
            "involving_different_students",
            "equal_participation_opportunities",
            "welcoming_different_viewpoints",
        ],
    },
    "individual_attention": {
        "weight": 0.10,
        "sub_features": [
            "asking_individual_students",
            "responding_to_individual_doubts",
            "checking_individual_understanding",
        ],
    },
    "teacher_behaviour_attitude": {
        "weight": 0.10,
        "sub_features": [
            "respectful_behaviour",
            "patience",
            "supportive_attitude",
            "positive_body_language",
        ],
    },
    "rapport_building": {
        "weight": 0.08,
        "sub_features": [
            "building_trust",
            "approachability",
            "comfortable_learning_environment",
        ],
    },
    "questioning_feedback": {
        "weight": 0.08,
        "sub_features": [
            "relevant_questions",
            "follow_up_questions",
            "constructive_feedback",
            "respectful_error_correction",
        ],
    },
    "classroom_management": {
        "weight": 0.06,
        "sub_features": [
            "managing_disruptions",
            "maintaining_discipline",
            "smooth_transitions",
            "time_management",
        ],
    },
    "adaptability_responsiveness": {
        "weight": 0.06,
        "sub_features": [
            "clarifying_confusion",
            "adjusting_explanation",
            "adjusting_pace",
            "responding_to_doubts",
        ],
    },
    "motivation_encouragement": {
        "weight": 0.06,
        "sub_features": [
            "praising_effort",
            "encouraging_questions",
            "building_student_confidence",
        ],
    },
    "teaching_method_variety": {
        "weight": 0.06,
        "sub_features": [
            "using_examples",
            "demonstrations",
            "class_discussions",
            "practical_applications",
        ],
    },
    "student_understanding_checks": {
        "weight": 0.06,
        "sub_features": [
            "recapping_concepts",
            "comprehension_questions",
            "checking_responses",
            "summarising_key_points",
        ],
    },
}


def display_teaching_style_features():
    total_weight = sum(
        feature["weight"]
        for feature in TEACHING_STYLE_FEATURES.values()
    )

    print("TEACHING STYLE FEATURES")
    print("=" * 40)

    for name, details in TEACHING_STYLE_FEATURES.items():
        print(f"\n{name} ({details['weight']:.0%})")

        for sub_feature in details["sub_features"]:
            print(f"  - {sub_feature}")

    print("\nTotal main features:", len(TEACHING_STYLE_FEATURES))
    print(f"Total weight: {total_weight:.0%}")


if __name__ == "__main__":
    display_teaching_style_features()


