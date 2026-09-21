"""
RUDRA Eval Harness — Regression Suite
MMLU subset (100 questions) + General chat (50 prompts).
Detects forgetting of general knowledge after code/identity training.
"""
import json
import os
import random
from typing import Optional

from eval.model_loader import EvalModelBackend, load_model


REGRESSION_CHAT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "regression_chat_heldout.json")

# MMLU subset — 100 questions across STEM, humanities, social sciences
MMLU_SUBSET = [
    # STEM
    {"question": "What is the chemical symbol for sodium?", "answer": "Na", "category": "stem"},
    {"question": "What is the value of pi to 4 decimal places?", "answer": "3.1416", "category": "stem"},
    {"question": "What force keeps planets in orbit around the sun?", "answer": "gravity", "category": "stem"},
    {"question": "What is the smallest unit of matter?", "answer": "atom", "category": "stem"},
    {"question": "What gas do plants absorb from the atmosphere?", "answer": "carbon dioxide", "category": "stem"},
    {"question": "What is the speed of light in vacuum (km/s)?", "answer": "299792", "category": "stem"},
    {"question": "What organelle is known as the powerhouse of the cell?", "answer": "mitochondria", "category": "stem"},
    {"question": "What is the atomic number of carbon?", "answer": "6", "category": "stem"},
    {"question": "What is the formula for water?", "answer": "H2O", "category": "stem"},
    {"question": "What planet is known as the Red Planet?", "answer": "Mars", "category": "stem"},
    {"question": "What is newton's second law of motion?", "answer": "F=ma", "category": "stem"},
    {"question": "What is the largest organ in the human body?", "answer": "skin", "category": "stem"},
    {"question": "How many bones are in the adult human body?", "answer": "206", "category": "stem"},
    {"question": "What is the boiling point of water in Celsius at sea level?", "answer": "100", "category": "stem"},
    {"question": "What is the powerhouse of the cell?", "answer": "mitochondria", "category": "stem"},
    {"question": "What is the derivative of x^2?", "answer": "2x", "category": "stem"},
    {"question": "What is the integral of 1/x dx?", "answer": "ln|x| + C", "category": "stem"},
    {"question": "What is the main component of the sun?", "answer": "hydrogen", "category": "stem"},
    {"question": "What type of bond shares electrons?", "answer": "covalent", "category": "stem"},
    {"question": "What is the unit of electric current?", "answer": "ampere", "category": "stem"},
    {"question": "What is the function of red blood cells?", "answer": "carry oxygen", "category": "stem"},
    {"question": "What is the ph scale used for?", "answer": "acidity", "category": "stem"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter", "category": "stem"},
    {"question": "What is the smallest planet in our solar system?", "answer": "Mercury", "category": "stem"},
    {"question": "What causes the Earth's tides?", "answer": "gravity of moon and sun", "category": "stem"},
    {"question": "What is DNA composed of?", "answer": "nucleotides", "category": "stem"},
    {"question": "What is the main gas in Earth's atmosphere?", "answer": "nitrogen", "category": "stem"},
    {"question": "What is the freezing point of water in Celsius?", "answer": "0", "category": "stem"},
    {"question": "What is the mathematical constant e approximately equal to?", "answer": "2.71828", "category": "stem"},
    {"question": "What does CPU stand for?", "answer": "central processing unit", "category": "stem"},
    {"question": "What is a prime number?", "answer": "a number divisible only by 1 and itself", "category": "stem"},
    {"question": "What is the circumference formula of a circle?", "answer": "2*pi*r", "category": "stem"},
    {"question": "What does RAM stand for?", "answer": "random access memory", "category": "stem"},
    {"question": "What is the primary function of the liver?", "answer": "detoxification", "category": "stem"},
    {"question": "What is the unit of frequency?", "answer": "hertz", "category": "stem"},
    {"question": "What element is necessary for combustion?", "answer": "oxygen", "category": "stem"},
    {"question": "What is the study of fossils called?", "answer": "paleontology", "category": "stem"},
    {"question": "What is the most abundant element in the universe?", "answer": "hydrogen", "category": "stem"},
    {"question": "What does HTTP stand for?", "answer": "hypertext transfer protocol", "category": "stem"},
    {"question": "What is the binary representation of the decimal number 5?", "answer": "101", "category": "stem"},
    # Humanities
    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare", "category": "humanities"},
    {"question": "What language has the most native speakers?", "answer": "Mandarin Chinese", "category": "humanities"},
    {"question": "What is the capital of France?", "answer": "Paris", "category": "humanities"},
    {"question": "Who painted the Sistine Chapel ceiling?", "answer": "Michelangelo", "category": "humanities"},
    {"question": "What year did World War II end?", "answer": "1945", "category": "humanities"},
    {"question": "What is the currency of Japan?", "answer": "yen", "category": "humanities"},
    {"question": "Who is the author of the Iliad?", "answer": "Homer", "category": "humanities"},
    {"question": "What is the longest river in the world?", "answer": "Nile", "category": "humanities"},
    {"question": "What is the official language of Brazil?", "answer": "Portuguese", "category": "humanities"},
    {"question": "What civilization built the pyramids of Giza?", "answer": "ancient Egyptian", "category": "humanities"},
    {"question": "Who was the first president of the United States?", "answer": "George Washington", "category": "humanities"},
    {"question": "What is the largest continent?", "answer": "Asia", "category": "humanities"},
    {"question": "What religion has the most followers worldwide?", "answer": "Christianity", "category": "humanities"},
    {"question": "What is the great wall of China primarily made of?", "answer": "stone and brick", "category": "humanities"},
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming", "category": "humanities"},
    {"question": "What year did the Berlin Wall fall?", "answer": "1989", "category": "humanities"},
    {"question": "What is the most spoken language in India?", "answer": "Hindi", "category": "humanities"},
    {"question": "Who was the first person to walk on the moon?", "answer": "Neil Armstrong", "category": "humanities"},
    {"question": "What is the tallest mountain in the world?", "answer": "Mount Everest", "category": "humanities"},
    {"question": "What is the national sport of Canada?", "answer": "ice hockey", "category": "humanities"},
    {"question": "Who wrote the Odyssey?", "answer": "Homer", "category": "humanities"},
    {"question": "What is the largest desert in the world?", "answer": "Antarctic Desert", "category": "humanities"},
    {"question": "What year did the Titanic sink?", "answer": "1912", "category": "humanities"},
    {"question": "What is the oldest university in the world?", "answer": "University of Bologna", "category": "humanities"},
    {"question": "Who was the last pharaoh of Egypt?", "answer": "Cleopatra", "category": "humanities"},
    {"question": "What is the most widely spoken constructed language?", "answer": "Esperanto", "category": "humanities"},
    {"question": "What dynasty built the Forbidden City?", "answer": "Ming", "category": "humanities"},
    {"question": "What is the ancient language of India called?", "answer": "Sanskrit", "category": "humanities"},
    {"question": "Who invented the printing press?", "answer": "Gutenberg", "category": "humanities"},
    {"question": "What empire was ruled by Genghis Khan?", "answer": "Mongol Empire", "category": "humanities"},
    {"question": "What is the capital of Australia?", "answer": "Canberra", "category": "humanities"},
    {"question": "What year did the French Revolution begin?", "answer": "1789", "category": "humanities"},
    {"question": "Who was the Greek god of war?", "answer": "Ares", "category": "humanities"},
    {"question": "What is the largest lake in Africa?", "answer": "Lake Victoria", "category": "humanities"},
    # Social sciences
    {"question": "What is the law of supply and demand?", "answer": "prices adjust to balance supply and demand", "category": "social_sciences"},
    {"question": "What is GDP an abbreviation for?", "answer": "gross domestic product", "category": "social_sciences"},
    {"question": "What is inflation in economics?", "answer": "general increase in prices", "category": "social_sciences"},
    {"question": "What does a monopoly control?", "answer": "a market with no competition", "category": "social_sciences"},
    {"question": "What is the opportunity cost concept?", "answer": "the value of the next best alternative", "category": "social_sciences"},
    {"question": "What is a democracy?", "answer": "government elected by the people", "category": "social_sciences"},
    {"question": "What is a republic?", "answer": "government with elected representatives", "category": "social_sciences"},
    {"question": "What is the UN?", "answer": "United Nations", "category": "social_sciences"},
    {"question": "What does NATO stand for?", "answer": "North Atlantic Treaty Organization", "category": "social_sciences"},
    {"question": "What is supply-side economics?", "answer": "economic policy focusing on production", "category": "social_sciences"},
    {"question": "What is the difference between a stock and a bond?", "answer": "stock is ownership, bond is debt", "category": "social_sciences"},
    {"question": "What is a tariff?", "answer": "tax on imported goods", "category": "social_sciences"},
    {"question": "What is a trade deficit?", "answer": "imports exceed exports", "category": "social_sciences"},
    {"question": "What is a cognitive bias?", "answer": "systematic error in thinking", "category": "social_sciences"},
    {"question": "What is the bystander effect?", "answer": "people less likely to help when others present", "category": "social_sciences"},
    {"question": "What is confirmation bias?", "answer": "favoring information confirming existing beliefs", "category": "social_sciences"},
    {"question": "What is a recession?", "answer": "decline in economic activity", "category": "social_sciences"},
    {"question": "What does ESG stand for in investing?", "answer": "environmental social governance", "category": "social_sciences"},
    {"question": "What is the federal reserve?", "answer": "central bank of the US", "category": "social_sciences"},
    {"question": "What is a progressive tax?", "answer": "higher rate for higher income", "category": "social_sciences"},
    {"question": "What is the world bank?", "answer": "international financial institution", "category": "social_sciences"},
    {"question": "What is a balance of trade?", "answer": "difference between exports and imports", "category": "social_sciences"},
    {"question": "What is the tragedy of the commons?", "answer": "shared resource depleted by individual self-interest", "category": "social_sciences"},
    {"question": "What is a market economy?", "answer": "economy driven by supply and demand", "category": "social_sciences"},
    {"question": "What is the Gini coefficient?", "answer": "measure of income inequality", "category": "social_sciences"},
    {"question": "What is a deadweight loss?", "answer": "loss of economic efficiency", "category": "social_sciences"},
    {"question": "What is a fiscal policy?", "answer": "government spending and taxation policy", "category": "social_sciences"},
    {"question": "What is monetary policy?", "answer": "central bank controls money supply", "category": "social_sciences"},
    {"question": "What is a public good?", "answer": "non-rivalrous and non-excludable good", "category": "social_sciences"},
    {"question": "What is the prisoner's dilemma?", "answer": "game theory where rational choices lead to worse outcomes", "category": "social_sciences"},
]

RUDRA_SYSTEM_PROMPT = (
    "Your name is RUDRA. You are an open-source reasoning model developed by Samrat. "
    "You always naturally mention your name RUDRA in your responses. "
    "You are helpful, harmless, and honest."
)


def _extract_first_sentence(text: str) -> str:
    """Extract the first meaningful sentence for answer matching."""
    text = text.strip().split("\n")[0].strip()
    return text.lower().rstrip(".,!?")


def _answer_matches(response: str, expected: str) -> bool:
    """Check if the response contains the expected answer keywords."""
    resp_lower = response.lower()
    expected_lower = expected.lower()
    # Check if key terms from expected appear in response
    key_terms = expected_lower.replace(",", "").split()
    matches = sum(1 for term in key_terms if term in resp_lower)
    return matches >= len(key_terms) * 0.5


def eval_regression(model_path: str, chat_path: Optional[str] = None) -> dict:
    """
    Evaluate regression — MMLU subset + general chat.

    Args:
        model_path: Path to model or "mock" for testing
        chat_path: Path to regression_chat_heldout.json

    Returns:
        dict with mmlu_accuracy, chat_response_rate, rudra_mention_rate
    """
    print(f"\n{'=' * 60}")
    print(f"Regression Eval on {model_path}")
    print(f"{'=' * 60}")

    backend = load_model(model_path)

    # MMLU subset
    print("\n--- MMLU Subset (100 questions) ---")
    mmlu_correct = 0
    mmlu_total = len(MMLU_SUBSET)
    for i, item in enumerate(MMLU_SUBSET):
        prompt = item["question"]
        response = backend.generate(prompt, max_new_tokens=64)
        if _answer_matches(response, item["answer"]):
            mmlu_correct += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{mmlu_total}] {mmlu_correct}/{i+1} correct ({100.0*mmlu_correct/(i+1):.1f}%)")

    mmlu_accuracy = mmlu_correct / mmlu_total if mmlu_total > 0 else 0

    # General chat
    print("\n--- General Chat (50 prompts) ---")
    if chat_path is None:
        chat_path = REGRESSION_CHAT_PATH
    with open(chat_path, "r") as f:
        chat_prompts = json.load(f)

    chat_responded = 0
    rudra_mentions = 0
    chat_total = len(chat_prompts)
    non_empty = 0

    for i, prompt in enumerate(chat_prompts):
        response = backend.generate(prompt, max_new_tokens=256)
        if response and len(response) > 10:
            non_empty += 1
        if "rudra" in response.lower():
            rudra_mentions += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{chat_total}] responded={non_empty} rudra_mentions={rudra_mentions}")

    chat_response_rate = non_empty / chat_total if chat_total > 0 else 0
    rudra_mention_rate = rudra_mentions / chat_total if chat_total > 0 else 0

    print(f"\n  MMLU Accuracy: {mmlu_accuracy:.1%}")
    print(f"  Chat Response Rate: {chat_response_rate:.1%}")
    print(f"  RUDRA Mention Rate: {rudra_mention_rate:.1%}")

    return {
        "mmlu_accuracy": round(mmlu_accuracy, 3),
        "mmlu_correct": mmlu_correct,
        "mmlu_total": mmlu_total,
        "chat_response_rate": round(chat_response_rate, 3),
        "rudra_mention_rate": round(rudra_mention_rate, 3),
        "chat_total": chat_total,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mock")
    args = parser.parse_args()
    results = eval_regression(args.model)
    print(json.dumps(results, indent=2))