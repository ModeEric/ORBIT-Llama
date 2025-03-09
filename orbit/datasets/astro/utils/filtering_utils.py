"""
Astronomy Filtering Utilities

This module provides tools for filtering and processing astronomy-specific content.
"""

import os
import json
import re
from typing import List, Dict, Any, Set, Optional
from tqdm import tqdm

class AstroFilter:
    """
    Specialized filter for astronomy content.
    
    This class provides methods for extracting astronomy-specific concepts
    and filtering astronomy content.
    """
    
    def __init__(self, custom_keywords: List[str] = None):
        """
        Initialize the AstroFilter.
        
        Args:
            custom_keywords: Additional astronomy keywords to include
        """
        # Load default astronomy keywords
        self.keywords = [
            "galaxy", "star", "planet", "asteroid", "comet", "nebula",
            "supernova", "quasar", "pulsar", "black hole", "telescope",
            "observatory", "cosmos", "universe", "solar system", "moon",
            "mars", "jupiter", "saturn", "venus", "mercury", "uranus",
            "neptune", "pluto", "exoplanet", "constellation", "meteor",
            "cosmic", "orbit", "gravity", "light-year", "parsec",
            "astronomer", "astrophysics", "cosmology", "astronomy",
            "stellar", "galactic", "interstellar", "planetary",
            "satellite", "space", "nasa", "esa", "hubble", "james webb",
            "chandra", "spitzer", "kepler", "voyager", "cassini",
            "dark matter", "dark energy", "big bang", "redshift",
            "gravitational wave", "neutron star", "white dwarf",
            "red giant", "main sequence", "spectroscopy", "parallax"
        ]
        
        # Add custom keywords if provided
        if custom_keywords:
            self.keywords.extend(custom_keywords)
        
        # Remove duplicates and sort
        self.keywords = sorted(set(self.keywords))
        
        # Compile regex patterns for faster matching
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for keyword matching."""
        # Create case-insensitive word boundary patterns for each keyword
        self.patterns = [
            re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            for kw in self.keywords
        ]
    
    def extract_astronomy_concepts(self, text: str) -> List[str]:
        """
        Extract astronomy concepts from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of astronomy concepts found in the text
        """
        found_concepts = set()
        
        # Check for each keyword pattern
        for i, pattern in enumerate(self.patterns):
            if pattern.search(text):
                found_concepts.add(self.keywords[i])
        
        return sorted(found_concepts)
    
    def filter_astronomy_content(self, text: str, min_concepts: int = 2) -> bool:
        """
        Filter text based on astronomy content.
        
        Args:
            text: Text to filter
            min_concepts: Minimum number of astronomy concepts required
            
        Returns:
            True if text passes the filter, False otherwise
        """
        concepts = self.extract_astronomy_concepts(text)
        return len(concepts) >= min_concepts
    
    def is_astronomy_content(self, text: str, 
                            custom_keywords: Optional[List[str]] = None) -> bool:
        """
        Determine if text is astronomy-related.
        
        Args:
            text: Text to analyze
            custom_keywords: Additional keywords to check for
            
        Returns:
            True if text is astronomy-related, False otherwise
        """
        # Extract concepts
        concepts = self.extract_astronomy_concepts(text)
        
        # Check custom keywords if provided
        if custom_keywords:
            custom_pattern = r'\b(' + '|'.join(re.escape(kw) for kw in custom_keywords) + r')\b'
            custom_matches = re.findall(custom_pattern, text, re.IGNORECASE)
            concepts.extend(custom_matches)
        
        # Calculate keyword density
        word_count = len(text.split())
        keyword_density = len(concepts) / max(1, word_count)
        
        # Check criteria
        return (len(concepts) >= 2 and 
                keyword_density >= 0.01)
    
    def filter_file(self, input_path: str, output_path: str,
                   custom_keywords: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Filter a file for astronomy content.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            custom_keywords: Additional keywords to check for
            
        Returns:
            Statistics about the filtering process
        """
        stats = {"total": 0, "kept": 0}
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, desc=f"Filtering {os.path.basename(input_path)}"):
                stats["total"] += 1
                
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    
                    if self.is_astronomy_content(text, custom_keywords):
                        outfile.write(line)
                        stats["kept"] += 1
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"Kept {stats['kept']}/{stats['total']} documents ({stats['kept']/max(1, stats['total'])*100:.1f}%)")
        return stats
    
    def analyze_astronomy_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze astronomy text for key metrics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        # Extract concepts
        concepts = self.extract_astronomy_concepts(text)
        
        # Calculate metrics
        word_count = len(text.split())
        keyword_density = len(concepts) / max(1, word_count)
        
        # Find most common concepts (if any appear multiple times)
        concept_counts = {}
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        top_concepts = sorted(
            concept_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]  # Top 10
        
        return {
            "word_count": word_count,
            "concept_count": len(concepts),
            "unique_concept_count": len(concept_counts),
            "keyword_density": keyword_density,
            "is_astronomy_content": self.is_astronomy_content(text),
            "top_concepts": top_concepts,
            "all_concepts": concepts
        } 