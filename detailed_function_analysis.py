#!/usr/bin/env python3
"""
Detailed function-by-function analysis of app.py
"""

import re

def extract_functions(content):
    """Extract all functions with their line numbers"""
    functions = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        if line.startswith('def ') or line.startswith('async def '):
            # Get function name
            match = re.match(r'(async )?def ([a-zA-Z_][a-zA-Z0-9_]*)\(', line)
            if match:
                func_name = match.group(2)
                is_async = match.group(1) is not None
                
                # Find the end of the function (next def or end of file)
                func_start = i
                func_end = len(lines)
                
                for j in range(i+1, len(lines)):
                    if lines[j-1].startswith('def ') or lines[j-1].startswith('async def '):
                        func_end = j - 1
                        break
                
                func_body = '\n'.join(lines[func_start-1:func_end])
                functions.append({
                    'name': func_name,
                    'async': is_async,
                    'start_line': func_start,
                    'end_line': func_end,
                    'body': func_body
                })
    
    return functions

def analyze_function(func):
    """Analyze a single function for issues"""
    issues = []
    warnings = []
    name = func['name']
    body = func['body']
    
    # Skip certain utility functions
    if name in ['get_home', 'health_check', 'gpu_diagnostics', 'cache_stats']:
        return issues, warnings
    
    print(f"\nAnalyzing {name} (lines {func['start_line']}-{func['end_line']})...")
    
    # 1. Check for model type handling
    if 'model_config["type"]' in body or "model_config['type']" in body:
        print(f"  → Uses model_config type checking")
        
        # Check if it handles faster_whisper
        if 'transcribe' in name.lower():
            has_faster_whisper = 'faster_whisper' in body
            has_openai = 'openai' in body and 'model_config["type"] == "openai"' in body
            has_deepgram = 'deepgram' in body and 'model_config["type"] == "deepgram"' in body
            
            if not has_faster_whisper and not has_deepgram:
                issues.append(f"{name}: Missing faster_whisper handling")
            
            # Check for empty blocks after GGML removal
            if 'elif model_config["type"] ==' in body:
                # Check if there are consecutive elif/else without content
                lines = body.split('\n')
                for i, line in enumerate(lines):
                    if 'elif model_config["type"] ==' in line:
                        # Check next few lines for content
                        next_lines = lines[i+1:min(i+5, len(lines))]
                        has_content = False
                        for nl in next_lines:
                            stripped = nl.strip()
                            if stripped and not stripped.startswith('#'):
                                has_content = True
                                break
                        if not has_content:
                            issues.append(f"{name}: Empty elif block at line {func['start_line'] + i}")
    
    # 2. Check model variable handling
    if 'model["model"]' in body or "model['model']" in body:
        print(f"  → Accesses model['model']")
        
        # Check if it's properly wrapped
        if 'isinstance(model, dict)' in body:
            print(f"    ✓ Has proper isinstance check")
        else:
            warnings.append(f"{name}: Accesses model['model'] without isinstance check")
    
    # 3. Check for processor.model usage
    if 'processor.model' in body:
        print(f"  → Uses processor.model")
        if name == 'transcribe_audio_stream':
            # This is expected for streaming
            print(f"    ✓ Expected in {name}")
    
    # 4. Check for MODEL_SIZE usage
    if 'MODEL_SIZE' in body and name not in ['health_check']:
        print(f"  → References MODEL_SIZE")
        
    # 5. Check for load_model calls
    if 'load_model(' in body:
        print(f"  → Calls load_model()")
        # Check what's being loaded
        load_pattern = r'load_model\(([^)]+)\)'
        matches = re.findall(load_pattern, body)
        for match in matches:
            print(f"    Loading: {match}")
            if 'ivrit-large-v3-turbo' in match:
                issues.append(f"{name}: Trying to load removed GGML model 'ivrit-large-v3-turbo'")
    
    # 6. Check for environment variables
    env_pattern = r'os\.getenv\("([^"]+)"'
    env_vars = re.findall(env_pattern, body)
    deprecated_envs = ['WHISPER_CPP_PATH', 'WHISPER_CPP_THREADS', 'IVRIT_MODEL_PATH']
    
    for var in env_vars:
        if var in deprecated_envs:
            issues.append(f"{name}: Uses deprecated env var '{var}'")
    
    # 7. Check for subprocess/ffmpeg usage
    if 'subprocess' in body or 'ffmpeg' in body:
        print(f"  → Uses subprocess/ffmpeg")
        if 'whisper-cli' in body or 'whisper.cpp' in body:
            issues.append(f"{name}: References whisper.cpp CLI")
    
    # 8. Check WebSocket functions for proper model handling
    if 'websocket' in func['body'] and 'transcribe' in name:
        print(f"  → WebSocket transcription function")
        
        # Check if it loads a model
        if 'load_model' in body:
            print(f"    ✓ Loads model")
        
        # Check if it sends appropriate messages
        if 'websocket.send_json' in body:
            print(f"    ✓ Sends WebSocket messages")
    
    # 9. Check for proper error handling
    if 'try:' in body:
        # Count try blocks vs except blocks
        try_count = body.count('try:')
        except_count = body.count('except')
        if try_count != except_count:
            warnings.append(f"{name}: Mismatched try/except blocks ({try_count} try, {except_count} except)")
    
    return issues, warnings

def main():
    """Main analysis function"""
    print("=" * 70)
    print("DETAILED FUNCTION-BY-FUNCTION ANALYSIS OF app.py")
    print("=" * 70)
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Extract all functions
    functions = extract_functions(content)
    print(f"\nFound {len(functions)} functions to analyze")
    
    all_issues = []
    all_warnings = []
    
    # Analyze each function
    for func in functions:
        issues, warnings = analyze_function(func)
        if issues:
            all_issues.extend([(func['name'], issue) for issue in issues])
        if warnings:
            all_warnings.extend([(func['name'], warning) for warning in warnings])
    
    # Report results
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    if all_issues:
        print("\n❌ CRITICAL ISSUES:")
        for func_name, issue in all_issues:
            print(f"  • {issue}")
    else:
        print("\n✅ No critical issues found!")
    
    if all_warnings:
        print("\n⚠️  WARNINGS:")
        for func_name, warning in all_warnings:
            print(f"  • {warning}")
    else:
        print("\n✅ No warnings found!")
    
    # Special checks
    print("\n" + "=" * 70)
    print("SPECIAL CHECKS")
    print("=" * 70)
    
    # Check MODEL_SIZE default
    if 'MODEL_SIZE = os.getenv("WHISPER_MODEL", "ivrit-ct2")' in content:
        print("✅ Default model is correctly set to ivrit-ct2")
    else:
        print("❌ Default model is not set to ivrit-ct2")
        all_issues.append(("global", "Default model not set to ivrit-ct2"))
    
    # Check if MODEL_CONFIGS includes ivrit-ct2
    if '"ivrit-ct2":' in content:
        print("✅ ivrit-ct2 is in MODEL_CONFIGS")
    else:
        print("❌ ivrit-ct2 missing from MODEL_CONFIGS")
        all_issues.append(("global", "ivrit-ct2 missing from MODEL_CONFIGS"))
    
    return len(all_issues) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)