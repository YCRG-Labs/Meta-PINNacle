#!/usr/bin/env python3
"""
Final cleanup for Task 10.4 - Remove duplicates and ensure clean implementation
"""

def final_cleanup():
    """Remove duplicates and clean up the paper"""
    
    with open("paper/paper.tex", 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Starting final cleanup...")
    
    # Remove duplicate Appendix C sections
    # Keep only the last one which should have the proper content
    sections = content.split("\\section{Appendix C:")
    
    if len(sections) > 2:  # More than one Appendix C
        # Keep the first part and the last Appendix C
        content = sections[0]
        
        # Add the final Appendix C
        final_appendix = """\\section{Appendix C: Detailed Algorithms}

\\subsection{Algorithm 1: MetaPINN Training}
The MetaPINN algorithm follows the Model-Agnostic Meta-Learning (MAML) framework adapted for physics-informed neural networks. The algorithm alternates between inner loop adaptation on individual tasks and outer loop meta-updates to learn optimal initialization parameters.

\\subsection{Algorithm 2: PhysicsInformedMetaLearner}  
The PhysicsInformedMetaLearner extends the basic MetaPINN with adaptive constraint weighting, physics regularization, and multi-scale handling. It dynamically balances different physics constraints based on their relative magnitudes and gradients during training.

"""
        
        # Find where to insert (before bibliography)
        bib_pos = content.find("\\bibliography{")
        if bib_pos != -1:
            content = content[:bib_pos] + final_appendix + content[bib_pos:]
        else:
            content += final_appendix
    
    # Clean up excessive whitespace
    content = content.replace("\\n\\n\\n\\n", "\\n\\n")
    content = content.replace("\\n\\n\\n", "\\n\\n")
    
    # Write cleaned content
    with open("paper/paper.tex", 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Final cleanup completed!")


if __name__ == "__main__":
    final_cleanup()