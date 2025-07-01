#!/usr/bin/env python3
"""
Script to find and fix all dict_keys pickle issues in the VAD codebase
"""
import os
import re
import sys

def find_dict_keys_issues(directory):
    """Find all potential dict_keys assignments that could cause pickle errors"""
    issues = []
    
    # Pattern to match self.attr = something.keys()
    pattern1 = re.compile(r'(self\.\w+)\s*=\s*([^=\n]+)\.keys\(\)')
    # Pattern to match variable = something.keys() at module level
    pattern2 = re.compile(r'^(\w+)\s*=\s*([^=\n]+)\.keys\(\)', re.MULTILINE)
    
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        if '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py') and file != 'fix_dict_keys.py':
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check for pattern 1
                    for match in pattern1.finditer(content):
                        line_num = content[:match.start()].count('\n') + 1
                        if 'list(' not in match.group(0) and 'sorted(' not in match.group(0):
                            issues.append({
                                'file': filepath,
                                'line': line_num,
                                'text': match.group(0),
                                'type': 'attribute'
                            })
                    
                    # Check for pattern 2 (module level)
                    for match in pattern2.finditer(content):
                        line_num = content[:match.start()].count('\n') + 1
                        if 'list(' not in match.group(0) and 'sorted(' not in match.group(0):
                            # Check if it's at module level (not inside a function)
                            lines_before = content[:match.start()].split('\n')
                            indent_level = len(match.group(0)) - len(match.group(0).lstrip())
                            if indent_level == 0:  # Module level
                                issues.append({
                                    'file': filepath,
                                    'line': line_num,
                                    'text': match.group(0),
                                    'type': 'module'
                                })
                                
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    
    return issues

def main():
    directory = '/home/ubuntu/work/VAD_Universe'
    
    print("Searching for dict_keys issues...")
    issues = find_dict_keys_issues(directory)
    
    if not issues:
        print("No dict_keys issues found!")
    else:
        print(f"\nFound {len(issues)} potential dict_keys issues:\n")
        for issue in issues:
            print(f"File: {issue['file']}")
            print(f"Line {issue['line']}: {issue['text']}")
            print(f"Type: {issue['type']}")
            print("-" * 80)

if __name__ == "__main__":
    main()