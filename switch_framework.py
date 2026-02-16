#!/usr/bin/env python3
"""
Framework Switcher for Konkani ASR
Easily switch between different ASR frameworks
"""

import yaml
import argparse
import sys

def switch_framework(framework):
    """Switch the active ASR framework"""

    # Validate framework
    valid_frameworks = ["huggingface", "nemo", "ai4bharat"]
    if framework not in valid_frameworks:
        print(f"❌ Invalid framework: {framework}")
        print(f"Valid options: {', '.join(valid_frameworks)}")
        return False

    # Load current config
    config_path = "configs/main_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return False

    # Update framework
    old_framework = config['framework']
    config['framework'] = framework

    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("✅ Framework switched successfully!")
    print(f"   From: {old_framework.upper()}")
    print(f"   To: {framework.upper()}")
    print()
    print("Framework Details:")
    framework_info = config['frameworks'][framework]
    print(f"   Base Model: {framework_info['base_model']}")
    print(f"   Model Directory: {framework_info['model_dir']}")
    print(f"   Config File: {framework_info['config_file']}")

    return True

def list_frameworks():
    """List all available frameworks"""

    config_path = "configs/main_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return

    current_framework = config['framework']

    print("🎯 Available ASR Frameworks:")
    print("=" * 40)

    for framework_name, framework_config in config['frameworks'].items():
        status = "✅ ACTIVE" if framework_name == current_framework else "   "
        print(f"{status} {framework_name.upper()}")
        print(f"      Base Model: {framework_config['base_model']}")
        print(f"      Output Dir: {framework_config['model_dir']}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Switch ASR frameworks for Konkani ASR")
    parser.add_argument("framework", nargs="?", help="Framework to switch to (huggingface, nemo, ai4bharat)")
    parser.add_argument("--list", "-l", action="store_true", help="List all available frameworks")

    args = parser.parse_args()

    if args.list:
        list_frameworks()
        return

    if not args.framework:
        print("❌ Please specify a framework or use --list to see options")
        print()
        print("Usage examples:")
        print("  python switch_framework.py huggingface")
        print("  python switch_framework.py --list")
        sys.exit(1)

    if switch_framework(args.framework):
        print()
        print("🚀 Ready to run fine-tuning with the new framework!")
        print(f"   Run: python scripts/fine_tune_{args.framework}.py")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()