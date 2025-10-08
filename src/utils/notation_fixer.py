#!/usr/bin/env python3
"""
Notation consistency fixer for paper revision.
Fixes θ/ξ parameter notation throughout the paper.
"""

import re
import os
from typing import List, Tuple, Dict

class NotationFixer:
    """Fixes notation consistency in LaTeX papers"""
    
    def __init__(self):
        # Patterns for PDE parameters (should use ξ)
        self.pde_parameter_patterns = [
            # Direct parameter references in equations
            (r'\\theta(?=\s*\\in)', r'\\xi'),  # θ ∈ Θ -> ξ ∈ Ξ
            (r'\\Theta(?=\s*\\subset)', r'\\Xi'),  # Θ ⊂ R^p -> Ξ ⊂ R^p
            (r'\\theta_\{?new\}?', r'\\xi_{new}'),  # θ_new -> ξ_new
            (r'\\theta_\{?i\}?', r'\\xi_i'),  # θ_i -> ξ_i
            (r'\\theta\]\s*=\s*0', r'\\xi] = 0'),  # F[u; θ] = 0 -> F[u; ξ] = 0
            (r'\\theta\)', r'\\xi)'),  # f(x; θ) -> f(x; ξ)
            (r'\\theta\]', r'\\xi]'),  # F[u; θ] -> F[u; ξ]
            (r'parameter vector \\theta', r'parameter vector \\xi'),
            (r'parameter configuration \\theta', r'parameter configuration \\xi'),
            (r'parameters \\theta', r'parameters \\xi'),
            (r'with \\theta', r'with \\xi'),
            # Parameter ranges and distributions
            (r'\\alpha \\in \[([0-9.]+), ([0-9.]+)\]', r'\\alpha \\in [\\1, \\2]'),  # Keep α as is
            (r'\\nu \\in \[([0-9.]+), ([0-9.]+)\]', r'\\nu \\in [\\1, \\2]'),  # Keep ν as is
            (r'Re \\in \[([0-9]+), ([0-9]+)\]', r'Re \\in [\\1, \\2]'),  # Keep Re as is
        ]
        
        # Patterns for neural network parameters (should use θ)
        self.network_parameter_patterns = [
            # Ensure network parameters use θ
            (r'neural network parameters \\phi', r'neural network parameters \\theta'),
            (r'network parameters \\phi', r'network parameters \\theta'),
            (r'\\min_\{?\\phi\}?', r'\\min_{\\theta}'),
            (r'\\nabla_\{?\\phi\}?', r'\\nabla_{\\theta}'),
            (r'\\phi_i\^\{?\([0-9k+]+\)\}?', lambda m: m.group(0).replace('\\phi', '\\theta')),
        ]
        
        # Context-specific fixes
        self.context_fixes = [
            # Ensure consistency in specific contexts
            (r'(parametric families of PDEs.*?)\\theta', r'\\1\\xi'),
            (r'(PDE parameter.*?)\\theta', r'\\1\\xi'),
            (r'(parameter configuration.*?)\\theta', r'\\1\\xi'),
            (r'(parameter vector.*?)\\theta', r'\\1\\xi'),
        ]
    
    def fix_notation_in_text(self, text: str) -> Tuple[str, List[str]]:
        """Fix notation in text and return fixed text with change log"""
        changes = []
        original_text = text
        
        # Apply PDE parameter fixes
        for pattern, replacement in self.pde_parameter_patterns:
            if isinstance(replacement, str):
                new_text = re.sub(pattern, replacement, text)
                if new_text != text:
                    matches = re.findall(pattern, text)
                    changes.append(f"PDE param: {pattern} -> {replacement} ({len(matches)} occurrences)")
                    text = new_text
        
        # Apply network parameter fixes
        for pattern, replacement in self.network_parameter_patterns:
            if isinstance(replacement, str):
                new_text = re.sub(pattern, replacement, text)
                if new_text != text:
                    matches = re.findall(pattern, text)
                    changes.append(f"Network param: {pattern} -> {replacement} ({len(matches)} occurrences)")
                    text = new_text
        
        # Apply context-specific fixes
        for pattern, replacement in self.context_fixes:
            new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if new_text != text:
                matches = re.findall(pattern, text, flags=re.IGNORECASE)
                changes.append(f"Context fix: {pattern} -> {replacement} ({len(matches)} occurrences)")
                text = new_text
        
        return text, changes
    
    def fix_paper_notation(self, paper_path: str) -> Dict[str, List[str]]:
        """Fix notation in paper file"""
        with open(paper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_content, changes = self.fix_notation_in_text(content)
        
        # Write back the fixed content
        with open(paper_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        return {'changes': changes, 'file': paper_path}
    
    def validate_notation_consistency(self, text: str) -> Dict[str, List[str]]:
        """Validate notation consistency and report issues"""
        issues = {
            'pde_param_issues': [],
            'network_param_issues': [],
            'mixed_usage': []
        }
        
        # Check for remaining θ in PDE contexts
        pde_contexts = [
            r'parametric.*?\\theta',
            r'PDE.*?\\theta',
            r'parameter vector.*?\\theta',
            r'\\mathcal\{F\}.*?\\theta'
        ]
        
        for pattern in pde_contexts:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                issues['pde_param_issues'].extend(matches)
        
        # Check for φ in network contexts
        network_contexts = [
            r'neural network.*?\\phi',
            r'network parameters.*?\\phi',
            r'\\min_.*?\\phi'
        ]
        
        for pattern in network_contexts:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                issues['network_param_issues'].extend(matches)
        
        return issues

def main():
    """Main function to fix notation in paper"""
    fixer = NotationFixer()
    
    # Fix notation in main paper
    paper_path = "paper/paper.tex"
    if os.path.exists(paper_path):
        print(f"Fixing notation in {paper_path}...")
        results = fixer.fix_paper_notation(paper_path)
        
        print(f"\nChanges made in {results['file']}:")
        for change in results['changes']:
            print(f"  - {change}")
        
        # Validate the fixes
        with open(paper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        issues = fixer.validate_notation_consistency(content)
        
        print(f"\nValidation results:")
        if any(issues.values()):
            print("  Issues found:")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    print(f"    {issue_type}: {len(issue_list)} issues")
                    for issue in issue_list[:3]:  # Show first 3
                        print(f"      - {issue}")
        else:
            print("  ✓ No notation consistency issues found")
    
    else:
        print(f"Paper file not found: {paper_path}")

if __name__ == "__main__":
    main()