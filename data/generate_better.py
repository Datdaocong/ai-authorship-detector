import csv
import os
import random

random.seed(42)

human_casual_templates = [
    "I woke up {time_phrase} and {action}.",
    "My {person} told me {small_event} before dinner.",
    "We spent the {time_unit} {activity} and talking about {topic}.",
    "I felt {emotion} when I saw {object} on the table.",
    "The {place} was {adjective} and smelled like {smell}.",
    "I found {object} while cleaning the {place}.",
    "I smiled when I remembered {memory}.",
    "The bus arrived late so I {action}.",
]

human_technical_templates = [
    "I spent the evening studying {technical_topic} for class.",
    "My professor explained how {technical_topic} works during the lecture.",
    "I tried implementing a simple {ml_task} but made a few mistakes.",
    "I read an article about {technical_topic} and took notes afterward.",
    "We discussed {technical_topic} in the lab this morning.",
    "I practiced writing code for {ml_task} last night.",
    "I reviewed examples of {technical_topic} before the quiz.",
    "I am still trying to understand how {technical_topic} is applied in practice.",
]

ai_casual_templates = [
    "You may notice that {ai_concept} can help with {benefit}.",
    "In many cases {ai_concept} is used to improve {domain}.",
    "Sometimes automated systems are designed to make {benefit} easier.",
    "Modern tools can support {domain} in surprisingly practical ways.",
    "AI based systems can often simplify {benefit}.",
    "Digital tools are becoming more useful in everyday {domain}.",
    "Many organizations now rely on intelligent systems for {benefit}.",
    "It is common for software platforms to use AI for {benefit}.",
]

ai_technical_templates = [
    "Machine learning models are commonly used for {ml_task}.",
    "Artificial intelligence systems can improve {domain} through {benefit_noun}.",
    "Natural language processing enables machines to {task}.",
    "Predictive models identify {insight_type} in data.",
    "Classification systems are often evaluated using {metric}.",
    "Large language models can generate {content_type} from prompts.",
    "Neural networks are effective for many {task_type} tasks.",
    "Data driven pipelines usually include preprocessing training and {evaluation_part}.",
]

time_phrases = [
    "earlier than usual", "late in the morning", "before class", "after lunch"
]

actions = [
    "made coffee", "checked my notes", "opened the window", "looked for my notebook",
    "sat quietly for a while", "wrote down a few ideas"
]

small_events = [
    "to bring an umbrella", "that the shop was closed", "to call home later",
    "to finish my homework first"
]

time_units = ["afternoon", "evening", "morning", "weekend"]

activities = [
    "studying in the library", "walking around the market", "cleaning the room",
    "waiting at the station", "cooking together"
]

topics = ["school", "old memories", "travel plans", "the future", "a strange dream"]

emotions = ["nervous", "relieved", "curious", "surprised", "peaceful", "tired"]

objects = [
    "an old notebook", "my headphones", "a photo album", "a handwritten letter",
    "a small gift", "a cracked mug"
]

places = ["kitchen", "hallway", "garden", "classroom", "living room", "balcony"]

adjectives = ["quiet", "warm", "peaceful", "a bit messy", "brighter than usual"]

smells = ["coffee", "tea", "old books", "fresh bread", "rain"]

memories = [
    "my first day at school", "an old conversation", "last summer",
    "a childhood friend", "that awkward moment yesterday"
]

technical_topics = [
    "machine learning", "neural networks", "text classification",
    "artificial intelligence", "data preprocessing", "feature extraction"
]

ml_tasks = [
    "classifier", "prediction model", "text analysis script",
    "training pipeline", "evaluation function"
]

ai_concepts = [
    "artificial intelligence", "machine learning", "automation",
    "language models", "intelligent systems"
]

benefits = [
    "decision making", "daily work", "content generation",
    "data analysis", "workflow automation", "information retrieval"
]

domains = [
    "education", "finance", "healthcare", "customer service",
    "software development", "retail"
]

benefit_nouns = [
    "automation", "optimization", "better forecasting",
    "data analysis", "decision support"
]

tasks = [
    "analyze text", "classify documents", "detect patterns",
    "generate summaries", "predict outcomes"
]

insight_types = [
    "trends", "anomalies", "useful patterns", "risk signals"
]

metrics = ["accuracy", "precision", "recall", "f1 score"]

content_types = [
    "summaries", "reports", "responses", "structured content"
]

task_types = [
    "classification", "prediction", "recommendation", "detection"
]

evaluation_parts = [
    "validation", "error analysis", "testing", "performance monitoring"
]

people = ["friend", "teacher", "neighbor", "uncle", "sister", "brother"]


def fill_human_casual():
    template = random.choice(human_casual_templates)
    return template.format(
        time_phrase=random.choice(time_phrases),
        action=random.choice(actions),
        person=random.choice(people),
        small_event=random.choice(small_events),
        time_unit=random.choice(time_units),
        activity=random.choice(activities),
        topic=random.choice(topics),
        emotion=random.choice(emotions),
        object=random.choice(objects),
        place=random.choice(places),
        adjective=random.choice(adjectives),
        smell=random.choice(smells),
        memory=random.choice(memories),
    )


def fill_human_technical():
    template = random.choice(human_technical_templates)
    return template.format(
        technical_topic=random.choice(technical_topics),
        ml_task=random.choice(ml_tasks),
    )


def fill_ai_casual():
    template = random.choice(ai_casual_templates)
    return template.format(
        ai_concept=random.choice(ai_concepts),
        benefit=random.choice(benefits),
        domain=random.choice(domains),
    )


def fill_ai_technical():
    template = random.choice(ai_technical_templates)
    return template.format(
        ml_task=random.choice(ml_tasks),
        domain=random.choice(domains),
        benefit_noun=random.choice(benefit_nouns),
        task=random.choice(tasks),
        insight_type=random.choice(insight_types),
        metric=random.choice(metrics),
        content_type=random.choice(content_types),
        task_type=random.choice(task_types),
        evaluation_part=random.choice(evaluation_parts),
    )


def generate_dataset():
    rows = []

    for _ in range(75):
        rows.append((fill_human_casual(), "human"))
        rows.append((fill_human_technical(), "human"))
        rows.append((fill_ai_casual(), "ai"))
        rows.append((fill_ai_technical(), "ai"))

    random.shuffle(rows)
    return rows


def main():
    output_path = os.path.join(os.path.dirname(__file__), "dataset_better.csv")
    rows = generate_dataset()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Saved dataset to: {output_path}")
    print(f"Total samples: {len(rows)}")


if __name__ == "__main__":
    main()