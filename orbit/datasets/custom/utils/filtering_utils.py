"""
Custom Domain Filtering Utilities

This module provides tools for filtering and processing content for custom domains.
"""

import os
import json
import re
from typing import List, Dict, Any, Set, Optional
from tqdm import tqdm

class CustomFilter:
    """
    Specialized filter for custom domain content.
    
    This class provides methods for extracting domain-specific concepts
    and filtering content based on user-defined keywords.
    """
    
    def __init__(self, domain_name: str, keywords: List[str]):
        """
        Initialize the CustomFilter.
        
        Args:
            domain_name: Name of the custom domain
            keywords: List of domain-specific keywords
        """
        self.domain_name = domain_name
        self.keywords = keywords
        
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
    
    def extract_domain_concepts(self, text: str) -> List[str]:
        """
        Extract domain-specific concepts from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of domain concepts found in the text
        """
        found_concepts = set()
        
        # Check for each keyword pattern
        for i, pattern in enumerate(self.patterns):
            if pattern.search(text):
                found_concepts.add(self.keywords[i])
        
        return sorted(found_concepts)
    
    def filter_domain_content(self, text: str, min_concepts: int = 2) -> bool:
        """
        Filter text based on domain content.
        
        Args:
            text: Text to filter
            min_concepts: Minimum number of domain concepts required
            
        Returns:
            True if text passes the filter, False otherwise
        """
        concepts = self.extract_domain_concepts(text)
        return len(concepts) >= min_concepts
    
    def is_domain_content(self, text: str, 
                         additional_keywords: Optional[List[str]] = None) -> bool:
        """
        Determine if text is related to the custom domain.
        
        Args:
            text: Text to analyze
            additional_keywords: Additional keywords to check for
            
        Returns:
            True if text is domain-related, False otherwise
        """
        # Extract concepts
        concepts = self.extract_domain_concepts(text)
        
        # Check additional keywords if provided
        if additional_keywords:
            custom_pattern = r'\b(' + '|'.join(re.escape(kw) for kw in additional_keywords) + r')\b'
            custom_matches = re.findall(custom_pattern, text, re.IGNORECASE)
            concepts.extend(custom_matches)
        
        # Calculate keyword density
        word_count = len(text.split())
        keyword_density = len(concepts) / max(1, word_count)
        
        # Check criteria
        return (len(concepts) >= 2 and 
                keyword_density >= 0.01)
    
    def filter_file(self, input_path: str, output_path: str,
                   additional_keywords: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Filter a file for domain content.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            additional_keywords: Additional keywords to check for
            
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
                    
                    if self.is_domain_content(text, additional_keywords):
                        outfile.write(line)
                        stats["kept"] += 1
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"Kept {stats['kept']}/{stats['total']} documents ({stats['kept']/max(1, stats['total'])*100:.1f}%)")
        return stats 