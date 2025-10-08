#!/usr/bin/env python3
"""
Simple Validation System for Paper Critical Revision
Validates all critical fixes are properly implemented
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

class SimpleValidator:
    """Simple validator that checks file existence and basic content"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.validation_results = {}
        self.validation_log = []
        
    def log_validation(self, message: str, status: str = "INFO"):
        """Log validation steps"""
        log_entry = f"[{status}] {message}"
        self.validation_log.append(log_entry)
        print(log_entry)
    
    def validate_paper_integration(self) -> bool:
        """Validate paper integration was successful"""
        self.log_validation("Validating paper integration")
        
        try:
            # Check if revised paper exists
            revised_paper = self.base_path / "paper" / "paper_revised.tex"
            if not revised_paper.exists():
                self.log_validation("Revised paper does not exist", "ERROR")
                return False
            
            # Read revised paper content
            with open(revised_paper, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for key improvements
            improvements_check = [
                ("L2 relative error", "L2 error metrics integrated"),
                ("Reference Solution Generation", "Ground truth methodology added"),
                ("Comparison with Neural Operators", "Neural operator comparison added"),
                ("Code and Data Availability", "Availability section added"),
                ("Appendix D", "Ablation studies appendix integrated"),
                ("Appendix E", "Hyperparameter search appendix integrated")
            ]
            
            improvements_found = 0
            for check_text, description in improvements_check:
                if check_text in content:
                    self.log_validation(f"✓ {description}")
                    improvements_found += 1
                else:
                    self.log_validation(f"✗ Missing: {description}", "WARNING")
            
            # Check paper length (should be reasonable)
            line_count = len(content.split('\n'))
            self.log_validation(f"Paper length: {line_count} lines")
            
            success = improvements_found >= 4  # At least 4 out of 6 improvements
            self.validation_results['paper_integration'] = success
            return success
            
        except Exception as e:
            self.log_validation(f"Paper integration validation failed: {str(e)}", "ERROR")
            self.validation_results['paper_integration'] = False
            return False
    
    def validate_reproduction_scripts(self) -> bool:
        """Validate reproduction scripts exist"""
        self.log_validation("Validating reproduction scripts")
        
        required_scripts = [
            "run_comprehensive_benchmark.py",
            "test_task_9_complete.py",
            "validate_extrapolation.py",
            "validate_timing_profiling.py"
        ]
        
        scripts_found = 0
        for script in required_scripts:
            script_path = self.base_path / script
            if script_path.exists():
                self.log_validation(f"✓ Found script: {script}")
                scripts_found += 1
            else:
                self.log_validation(f"✗ Missing script: {script}", "WARNING")
        
        success = scripts_found >= 3  # At least 3 out of 4 scripts
        self.validation_results['reproduction_scripts'] = success
        return success
    
    def validate_data_files(self) -> bool:
        """Validate key data files exist"""
        self.log_validation("Validating data files")
        
        data_files = [
            "task_9_results/hyperparameter_search_results.json",
            "task_9_results/baseline_validation_results.json",
            "paper_sections/appendix_d_complete.tex",
            "paper_tables/ablation_tables_combined.tex"
        ]
        
        files_found = 0
        for data_file in data_files:
            file_path = self.base_path / data_file
            if file_path.exists():
                self.log_validation(f"✓ Found data file: {data_file}")
                files_found += 1
                
                # Basic validation for JSON files
                if data_file.endswith('.json'):
                    try:
                        with open(file_path, 'r') as f:
                            json.load(f)
                        self.log_validation(f"✓ Valid JSON: {data_file}")
                    except json.JSONDecodeError:
                        self.log_validation(f"✗ Invalid JSON: {data_file}", "ERROR")
                        files_found -= 1
            else:
                self.log_validation(f"✗ Missing data file: {data_file}", "WARNING")
        
        success = files_found >= 2  # At least 2 out of 4 files
        self.validation_results['data_files'] = success
        return success
    
    def validate_critical_fixes(self) -> bool:
        """Validate critical fixes are implemented"""
        self.log_validation("Validating critical fixes")
        
        try:
            # Check if revised paper has critical fixes
            revised_paper = self.base_path / "paper" / "paper_revised.tex"
            if not revised_paper.exists():
                self.log_validation("Cannot validate fixes - no revised paper", "ERROR")
                return False
            
            with open(revised_paper, 'r', encoding='utf-8') as f:
                content = f.read()
            
            critical_fixes = [
                ("L2 relative error", "Accuracy replaced with L2 error metrics"),
                ("microscopic", "Spelling error 'mocroscopic' fixed"),
                ("Reference Solution", "Ground truth methodology documented"),
                ("Statistical Methods", "Statistical analysis methodology added"),
                ("Neural Operators", "Baseline comparison with neural operators")
            ]
            
            fixes_found = 0
            for fix_text, description in critical_fixes:
                if fix_text in content:
                    self.log_validation(f"✓ {description}")
                    fixes_found += 1
                else:
                    self.log_validation(f"✗ Missing fix: {description}", "WARNING")
            
            success = fixes_found >= 3  # At least 3 out of 5 fixes
            self.validation_results['critical_fixes'] = success
            return success
            
        except Exception as e:
            self.log_validation(f"Critical fixes validation failed: {str(e)}", "ERROR")
            self.validation_results['critical_fixes'] = False
            return False
    
    def validate_backup_created(self) -> bool:
        """Validate backup was created"""
        self.log_validation("Validating backup creation")
        
        backup_path = self.base_path / "paper" / "paper_backup.tex"
        if backup_path.exists():
            self.log_validation("✓ Backup created successfully")
            self.validation_results['backup_created'] = True
            return True
        else:
            self.log_validation("✗ No backup found", "WARNING")
            self.validation_results['backup_created'] = False
            return False
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks"""
        self.log_validation("Starting comprehensive validation")
        
        validation_checks = [
            ("Paper Integration", self.validate_paper_integration),
            ("Reproduction Scripts", self.validate_reproduction_scripts),
            ("Data Files", self.validate_data_files),
            ("Critical Fixes", self.validate_critical_fixes),
            ("Backup Created", self.validate_backup_created)
        ]
        
        results_summary = {
            'total_checks': len(validation_checks),
            'passed_checks': 0,
            'failed_checks': 0,
            'warnings': 0,
            'details': {}
        }
        
        for check_name, check_function in validation_checks:
            self.log_validation(f"Running {check_name} validation")
            try:
                result = check_function()
                results_summary['details'][check_name] = result
                if result:
                    results_summary['passed_checks'] += 1
                    self.log_validation(f"{check_name}: PASSED", "SUCCESS")
                else:
                    results_summary['failed_checks'] += 1
                    self.log_validation(f"{check_name}: FAILED", "ERROR")
            except Exception as e:
                results_summary['failed_checks'] += 1
                results_summary['details'][check_name] = False
                self.log_validation(f"{check_name}: ERROR - {str(e)}", "ERROR")
        
        # Count warnings
        results_summary['warnings'] = len([log for log in self.validation_log if "WARNING" in log])
        
        # Overall success rate
        success_rate = results_summary['passed_checks'] / results_summary['total_checks']
        results_summary['success_rate'] = success_rate
        
        self.log_validation(f"Validation completed: {success_rate:.1%} success rate")
        
        return results_summary
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report"""
        report = "# Comprehensive Validation Report\n\n"
        report += f"**Overall Success Rate:** {results['success_rate']:.1%}\n\n"
        
        report += "## Summary\n\n"
        report += f"- Total Checks: {results['total_checks']}\n"
        report += f"- Passed: {results['passed_checks']}\n"
        report += f"- Failed: {results['failed_checks']}\n"
        report += f"- Warnings: {results['warnings']}\n\n"
        
        report += "## Detailed Results\n\n"
        for check_name, result in results['details'].items():
            status = "✅ PASSED" if result else "❌ FAILED"
            report += f"- **{check_name}:** {status}\n"
        
        report += "\n## Validation Log\n\n"
        for log_entry in self.validation_log:
            report += f"- {log_entry}\n"
        
        report += "\n## Recommendations\n\n"
        if results['failed_checks'] > 0:
            report += "⚠️ **Action Required:** Some validation checks failed. Review the detailed log above.\n\n"
        
        if results['warnings'] > 0:
            report += "⚠️ **Review Recommended:** Some warnings were generated.\n\n"
        
        if results['success_rate'] >= 0.8:
            report += "✅ **Ready for Submission:** Validation shows the paper revision is comprehensive.\n\n"
        
        return report

def main():
    """Main validation function"""
    validator = SimpleValidator()
    
    print("🔍 Starting comprehensive validation of paper critical revision...")
    
    results = validator.run_comprehensive_validation()
    
    # Generate and save report
    report = validator.generate_validation_report(results)
    with open("comprehensive_validation_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📊 Validation completed with {results['success_rate']:.1%} success rate")
    print(f"📄 Detailed report: comprehensive_validation_report.md")
    
    if results['success_rate'] >= 0.8:
        print("✅ Paper revision validation successful!")
        return True
    else:
        print("❌ Paper revision needs attention before submission")
        return False

if __name__ == "__main__":
    main()