"""
Law Filtering Utilities

This module provides tools for filtering and processing law-specific content.
"""

import os
import json
import re
from typing import List, Dict, Any, Set, Optional
from tqdm import tqdm

class LawFilter:
    """
    Specialized filter for law content.
    
    This class provides methods for extracting law-specific concepts
    and filtering legal content.
    """
    
    def __init__(self, custom_keywords: List[str] = None):
        """
        Initialize the LawFilter.
        
        Args:
            custom_keywords: Additional law keywords to include
        """
        # Load default law keywords
        self.keywords = [
            "court", "judge", "plaintiff", "defendant", "attorney", "lawyer",
            "legal", "law", "statute", "regulation", "jurisdiction", "verdict",
            "ruling", "appeal", "trial", "litigation", "lawsuit", "case law",
            "precedent", "judicial", "constitutional", "contract", "tort",
            "criminal", "civil", "prosecution", "defense", "evidence",
            "testimony", "witness", "jury", "settlement", "damages",
            "injunction", "motion", "pleading", "brief", "affidavit",
            "deposition", "discovery", "subpoena", "indictment", "complaint",
            "habeas corpus", "due process", "negligence", "liability",
            "malpractice", "arbitration", "mediation", "probate", "patent",
            "copyright", "trademark", "intellectual property", "corporate",
            "bankruptcy", "tax law", "administrative law", "environmental law",
            "international law", "maritime law", "immigration law", "family law",
            "supreme court", "appellate court", "district court", "state court",
            "federal court", "justice", "dissent", "concurrence", "opinion",
            "statute of limitations", "burden of proof", "reasonable doubt",
            "preponderance of evidence", "hearsay", "objection", "overruled",
            "sustained", "cross-examination", "direct examination"
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
    
    def extract_law_concepts(self, text: str) -> List[str]:
        """
        Extract law concepts from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of law concepts found in the text
        """
        found_concepts = set()
        
        # Check for each keyword pattern
        for i, pattern in enumerate(self.patterns):
            if pattern.search(text):
                found_concepts.add(self.keywords[i])
        
        return sorted(found_concepts)
    
    def filter_law_content(self, text: str, min_concepts: int = 2) -> bool:
        """
        Filter text based on law content.
        
        Args:
            text: Text to filter
            min_concepts: Minimum number of law concepts required
            
        Returns:
            True if text passes the filter, False otherwise
        """
        concepts = self.extract_law_concepts(text)
        return len(concepts) >= min_concepts
    
    def is_law_content(self, text: str, 
                      custom_keywords: Optional[List[str]] = None) -> bool:
        """
        Determine if text is law-related.
        
        Args:
            text: Text to analyze
            custom_keywords: Additional keywords to check for
            
        Returns:
            True if text is law-related, False otherwise
        """
        # Extract concepts
        concepts = self.extract_law_concepts(text)
        
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
        Filter a file for law content.
        
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
                    
                    if self.is_law_content(text, custom_keywords):
                        outfile.write(line)
                        stats["kept"] += 1
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"Kept {stats['kept']}/{stats['total']} documents ({stats['kept']/max(1, stats['total'])*100:.1f}%)")
        return stats 