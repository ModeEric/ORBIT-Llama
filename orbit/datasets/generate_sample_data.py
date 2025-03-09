#!/usr/bin/env python
"""
Generate sample astronomy data for ORBIT testing.

This script creates synthetic astronomy text samples that can be used
to test the ORBIT curation pipeline without requiring external data.
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any

# Sample astronomy topics
ASTRONOMY_TOPICS = [
    "black holes", "galaxies", "stars", "planets", "exoplanets",
    "nebulae", "supernovae", "dark matter", "dark energy", "cosmology",
    "telescopes", "space missions", "gravitational waves", "quasars",
    "neutron stars", "pulsars", "asteroids", "comets", "solar system",
    "interstellar medium", "stellar evolution", "galaxy formation"
]

# Sample astronomy terms
ASTRONOMY_TERMS = [
    "accretion disk", "active galactic nucleus", "albedo", "aphelion",
    "asteroid belt", "astronomical unit", "binary star", "black hole",
    "celestial sphere", "chromosphere", "cosmic microwave background",
    "cosmological constant", "dark energy", "dark matter", "dwarf planet",
    "ecliptic", "event horizon", "exoplanet", "galaxy cluster", "gravitational lensing",
    "Hubble constant", "interstellar medium", "Kuiper belt", "light-year",
    "magnetosphere", "main sequence", "nebula", "neutron star", "Oort cloud",
    "parsec", "photosphere", "planetary nebula", "pulsar", "quasar",
    "red giant", "redshift", "solar wind", "spectroscopy", "supernova",
    "white dwarf"
]

# Sample telescopes and missions
SPACE_MISSIONS = [
    "Hubble Space Telescope", "James Webb Space Telescope", "Chandra X-ray Observatory",
    "Spitzer Space Telescope", "Kepler Space Telescope", "TESS", "Voyager",
    "Cassini", "New Horizons", "Juno", "LIGO", "Very Large Array",
    "European Extremely Large Telescope", "Thirty Meter Telescope",
    "Square Kilometre Array", "Gaia", "Planck", "Herschel Space Observatory"
]

def generate_astronomy_text(
    min_length: int = 200,
    max_length: int = 800,
    quality: str = "high"
) -> str:
    """
    Generate synthetic astronomy text.
    
    Args:
        min_length: Minimum text length
        max_length: Maximum text length
        quality: Quality level ('high', 'medium', 'low')
        
    Returns:
        Generated text
    """
    # Select a random topic
    topic = random.choice(ASTRONOMY_TOPICS)
    
    # Generate title
    title = f"Understanding {topic.title()}"
    
    # Generate paragraphs
    paragraphs = []
    
    # Introduction
    intro_terms = random.sample(ASTRONOMY_TERMS, k=min(3, len(ASTRONOMY_TERMS)))
    intro = f"{title}\n\n{topic.title()} are one of the most fascinating subjects in astronomy. "
    intro += f"They involve the study of {intro_terms[0]}s and their relationship with {intro_terms[1]}s. "
    
    if quality == "high":
        intro += f"Recent advancements using the {random.choice(SPACE_MISSIONS)} have revolutionized our understanding of {topic}. "
        intro += f"This has led to new theories about the nature of {intro_terms[2]}s and their role in the universe."
    elif quality == "medium":
        intro += f"Scientists use {random.choice(SPACE_MISSIONS)} to study {topic}. "
        intro += f"This helps us learn more about {intro_terms[2]}s."
    else:  # low quality
        intro += f"They are very interesting and cool. Scientists study them a lot."
    
    paragraphs.append(intro)
    
    # Main content
    if quality == "high":
        # 2-4 detailed paragraphs for high quality
        num_paragraphs = random.randint(2, 4)
        for i in range(num_paragraphs):
            terms = random.sample(ASTRONOMY_TERMS, k=min(4, len(ASTRONOMY_TERMS)))
            mission = random.choice(SPACE_MISSIONS)
            
            para = f"One important aspect of {topic} research involves {terms[0]}s. "
            para += f"When we observe {topic} using {mission}, we can detect the presence of {terms[1]}s. "
            para += f"This is significant because it helps us understand the relationship between {terms[2]}s and {terms[3]}s. "
            para += f"Recent studies have shown that this relationship is crucial for understanding the evolution of {topic}."
            
            paragraphs.append(para)
    elif quality == "medium":
        # 1-2 simpler paragraphs for medium quality
        num_paragraphs = random.randint(1, 2)
        for i in range(num_paragraphs):
            terms = random.sample(ASTRONOMY_TERMS, k=min(2, len(ASTRONOMY_TERMS)))
            
            para = f"{topic.title()} are interesting because they contain {terms[0]}s. "
            para += f"Scientists study {terms[1]}s to learn more about {topic}. "
            para += f"This helps us understand the universe better."
            
            paragraphs.append(para)
    else:  # low quality
        # 1 poor quality paragraph
        para = f"{topic} are really amazing. "
        para += f"I think they are the best thing in space. "
        para += f"Scientists should study them more because they're cool."
        
        paragraphs.append(para)
    
    # Conclusion
    if quality == "high":
        conclusion = f"In conclusion, the study of {topic} continues to yield fascinating insights into the nature of our universe. "
        conclusion += f"As technology advances and new missions like {random.choice(SPACE_MISSIONS)} come online, "
        conclusion += f"we can expect even more discoveries that will deepen our understanding of {topic} and their role in cosmic evolution."
    elif quality == "medium":
        conclusion = f"In summary, {topic} are an important part of astronomy. "
        conclusion += f"Future research will help us learn more about them."
    else:  # low quality
        conclusion = f"So that's why {topic} are cool. Thanks for reading!"
    
    paragraphs.append(conclusion)
    
    # Join paragraphs
    text = "\n\n".join(paragraphs)
    
    # Ensure text length is within bounds
    while len(text) < min_length:
        # Add more content if too short
        extra_term = random.choice(ASTRONOMY_TERMS)
        extra_text = f" The relationship between {topic} and {extra_term}s is also worth exploring further."
        text += extra_text
    
    if len(text) > max_length:
        # Truncate if too long, but try to end at a sentence
        text = text[:max_length]
        last_period = text.rfind('.')
        if last_period > min_length:
            text = text[:last_period+1]
    
    return text

def generate_dataset(
    num_samples: int = 100,
    output_path: str = "domain_filtered_data.jsonl",
    high_quality_ratio: float = 0.6,
    medium_quality_ratio: float = 0.3
) -> None:
    """
    Generate a sample dataset of astronomy texts.
    
    Args:
        num_samples: Number of samples to generate
        output_path: Path to save the dataset
        high_quality_ratio: Ratio of high quality samples
        medium_quality_ratio: Ratio of medium quality samples
    """
    samples = []
    
    # Calculate number of samples for each quality level
    num_high = int(num_samples * high_quality_ratio)
    num_medium = int(num_samples * medium_quality_ratio)
    num_low = num_samples - num_high - num_medium
    
    print(f"Generating {num_high} high quality samples...")
    for _ in range(num_high):
        text = generate_astronomy_text(quality="high")
        samples.append({"text": text, "quality": 1, "quality_score": random.uniform(0.7, 1.0)})
    
    print(f"Generating {num_medium} medium quality samples...")
    for _ in range(num_medium):
        text = generate_astronomy_text(quality="medium")
        samples.append({"text": text, "quality": random.choice([0, 1]), "quality_score": random.uniform(0.4, 0.7)})
    
    print(f"Generating {num_low} low quality samples...")
    for _ in range(num_low):
        text = generate_astronomy_text(quality="low")
        samples.append({"text": text, "quality": 0, "quality_score": random.uniform(0.1, 0.4)})
    
    # Shuffle samples
    random.shuffle(samples)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Generated {len(samples)} samples and saved to {output_path}")

def main():
    """Generate sample astronomy data."""
    parser = argparse.ArgumentParser(description="Generate sample astronomy data")
    parser.add_argument("--output", "-o", default="domain_filtered_data.jsonl", 
                       help="Output file path")
    parser.add_argument("--samples", "-n", type=int, default=100, 
                       help="Number of samples to generate")
    parser.add_argument("--high-quality", type=float, default=0.6, 
                       help="Ratio of high quality samples")
    parser.add_argument("--medium-quality", type=float, default=0.3, 
                       help="Ratio of medium quality samples")
    parser.add_argument("--seed", "-s", type=int, default=42, 
                       help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Generate dataset
    generate_dataset(
        num_samples=args.samples,
        output_path=args.output,
        high_quality_ratio=args.high_quality,
        medium_quality_ratio=args.medium_quality
    )

if __name__ == "__main__":
    main() 