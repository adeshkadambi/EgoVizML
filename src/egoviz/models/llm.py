"""This module contains all code for the inference and evaluation of LLaVA 1.6 models on the ANS-SCI Home Dataset."""

import os
from concurrent.futures import ThreadPoolExecutor

import ollama
from tqdm import tqdm


def llava_inference(
    model_name: str, image_path: str, seed: int = 42, temperature: float = 0.0
):

    assert model_name in ["llava:7b", "llava:13b", "llava:34b"]

    # pull the model if it hasnt been done yet
    ollama.pull(model_name)

    prompt = """
    The provided image contains 6 frames uniformly sampled from a video in a 3x2 grid view. The video contains a person with stroke or spinal cord injury performing activities of daily living (ADL) in their home environment. 

    What ADL class is the person performing in the frames provided?

    A. Self-Feeding [Description: Setting up, arranging, and bringing food or fluid from the plate or cup to the mouth]
    B. Functional Mobility [Description: Moving from one position or place to another, specifically things like in-bed mobility, wheelchair or walker mobility, and transfers from wheelchair to bed or vice versa.]
    C. Grooming & Health Management [Description: Obtaining and using supplies; removing body hair (e.g., using razor, tweezers, lotion); applying and removing cosmetics; washing, drying, combing, styling, brushing, and trimming hair; caring for nails (hands and feet); caring for skin, ears, eyes, and nose; applying deodorant; cleaning mouth; brushing and flossing teeth; and removing, cleaning, and reinserting dental orthotics and prosthetics. Developing, managing, and maintaining routines for health and wellness promotion, such as physical fitness, nutrition, decreased health risk behaviors, and medication routines.]
    D. Communication Management [Description: Covers the exchange and interpretation of information using various communication tools and equipment (e.g., phone, laptop).]
    E. Home Establishment and Management [Description: Obtaining and maintaining personal and household possessions and environment (e.g., home, yard, garden, appliances, vehicles), including maintaining and repairing personal possessions (e.g., clothing, household items) and knowing how to seek help or whom to contact.]
    F. Meal Preparation and Cleanup [Description: Encompasses planning, preparing, serving meals, and cleaning up afterward.]
    G. Leisure & Other Activities [Description: Nonobligatory activity that is intrinsically motivated and engaged in during discretionary time, that is, time not committed to obligatory occupations or any of the aforementioned ADLs. Includes activities like video games, knitting, etc.]

    Answer with the option's letter from the given choices directly. Only pick one option. If you are unsure, please provide your best guess.
    """

    result = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt, "images": [image_path]}],
        options={"seed": seed, "temperature": temperature},
    )

    mapping = {
        "A": "self-feeding",
        "B": "functional-mobility",
        "C": "grooming-health-management",
        "D": "communication-management",
        "E": "home-management",
        "F": "meal-prep-cleanup",
        "G": "leisure",
    }

    return mapping[result["message"]["content"].strip()]


def llava_multi_image(dir_path: str, **kwargs):
    ground_truth = dir_path.split("\\")[-1]

    results = {}

    # Define a function to process each image
    def process_image(image):
        image_path = os.path.join(dir_path, image)
        prediction = llava_inference(image_path=image_path, **kwargs)
        results[image_path] = {"prediction": prediction, "ground_truth": ground_truth}

    # Use ThreadPoolExecutor to parallelize image processing
    with ThreadPoolExecutor() as executor:
        # Submit each image for processing
        futures = [
            executor.submit(process_image, image) for image in os.listdir(dir_path)
        ]

        # Wait for all tasks to complete
        for future in tqdm(futures, total=len(futures), desc="Processing Images"):
            future.result()

    return results
