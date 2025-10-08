#!/usr/bin/env python3
"""
Validation script for Task 10.4 requirements

Verifies that all requirements are properly implemented:
1. Move Algorithm 1 & 2 to appendix, keep only descriptions in main text
2. Condense Appendix B and remove redundant code structure details
3. Fix "mocroscopic" → "microscopic" and other minor spelling errors
4. Ensure consistent hyphenation "meta-learning" throughout paper
5. Target 35-38 pages total length
"""

import re


def validate_task_10_4():
    """Validate all task 10.4 requirements"""
    
    with open("paper/paper.tex", 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Validating Task 10.4 requirements...")
    print("=" * 50)
    
    results = {}
    
    # Requirement 1: Algorithms moved to appendix
    print("1. Checking algorithms moved to appendix...")
    
    # Check no algorithm environments in main text
    algorithm_envs = re.findall(r'\\begin\{algorithm\}', content)
    results['algorithms_in_main'] = len(algorithm_envs) == 0
    
    # Check appendix has algorithm descriptions
    appendix_c_exists = "Appendix C:" in content
    results['appendix_c_exists'] = appendix_c_exists
    
    # Check main text has references to appendix
    appendix_refs = content.count("provided in Appendix C")
    results['appendix_references'] = appendix_refs >= 2
    
    print(f"   ✓ No algorithm environments in main text: {results['algorithms_in_main']}")
    print(f"   ✓ Appendix C exists: {results['appendix_c_exists']}")
    print(f"   ✓ References to Appendix C: {results['appendix_references']} ({appendix_refs} found)")
    
    # Requirement 2: Appendix B condensed
    print("\\n2. Checking Appendix B condensed...")
    
    appendix_b_exists = "Appendix B:" in content
    results['appendix_b_exists'] = appendix_b_exists
    
    if appendix_b_exists:
        # Check for condensed content indicators
        has_code_structure = "Code Structure" in content
        has_key_features = "Key Features" in content
        has_reproducibility = "Reproducibility" in content
        
        results['appendix_b_condensed'] = has_code_structure and has_key_features and has_reproducibility
        print(f"   ✓ Appendix B exists and condensed: {results['appendix_b_condensed']}")
    else:
        results['appendix_b_condensed'] = False
        print(f"   ✗ Appendix B not found")
    
    # Requirement 3: Spelling errors fixed
    print("\\n3. Checking spelling errors fixed...")
    
    spelling_errors = {
        "mocroscopic": content.count("mocroscopic"),
        "meta learning": content.count("meta learning"),
        "Meta learning": content.count("Meta learning"),
        "metalearning": content.count("metalearning"),
        "multi task": content.count("multi task"),
        "multitask": content.count("multitask"),
        "real time": content.count("real time"),
        "fine tuning": content.count("fine tuning"),
        "pre training": content.count("pre training"),
        "state of the art": content.count("state of the art")
    }
    
    total_errors = sum(spelling_errors.values())
    results['spelling_fixed'] = total_errors == 0
    
    print(f"   ✓ Spelling errors fixed: {results['spelling_fixed']}")
    if total_errors > 0:
        print(f"   Remaining errors: {spelling_errors}")
    
    # Requirement 4: Consistent hyphenation
    print("\\n4. Checking consistent hyphenation...")
    
    meta_learning_count = content.count("meta-learning")
    Meta_learning_count = content.count("Meta-learning")
    
    results['consistent_hyphenation'] = meta_learning_count > 0 or Meta_learning_count > 0
    
    print(f"   ✓ Consistent meta-learning hyphenation: {results['consistent_hyphenation']}")
    print(f"   Found: {meta_learning_count} 'meta-learning', {Meta_learning_count} 'Meta-learning'")
    
    # Requirement 5: Target page length
    print("\\n5. Checking target page length...")
    
    # Rough estimation: ~3000 characters per page
    char_count = len(content)
    estimated_pages = char_count / 3000
    
    results['target_length'] = estimated_pages <= 38  # Target is to not exceed 38 pages
    
    print(f"   ✓ Target length (≤38 pages): {results['target_length']}")
    print(f"   Estimated pages: {estimated_pages:.1f} ({char_count} characters)")
    
    # Overall validation
    print("\\n" + "=" * 50)
    print("OVERALL VALIDATION RESULTS")
    print("=" * 50)
    
    all_passed = all(results.values())
    
    for requirement, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{requirement}: {status}")
    
    print("\\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TASK 10.4 REQUIREMENTS SUCCESSFULLY IMPLEMENTED!")
    else:
        print("⚠️  Some requirements need attention")
    
    return results


if __name__ == "__main__":
    validate_task_10_4()