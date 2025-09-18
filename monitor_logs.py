#!/usr/bin/env python3
"""
Log Monitoring Script for Trading Bot
"""
import os
import time
from pathlib import Path
from datetime import datetime

def monitor_logs(log_dir: str = "logs", follow: bool = True):
    """Monitor log files in real-time"""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"❌ Log directory {log_dir} does not exist!")
        return
    
    print(f"🔍 Monitoring logs in: {log_path.absolute()}")
    print("Press Ctrl+C to stop monitoring\n")
    
    # Find all log files
    log_files = list(log_path.glob("*.log"))
    
    if not log_files:
        print("📭 No log files found. Waiting for logs to be created...")
        while not log_files:
            time.sleep(1)
            log_files = list(log_path.glob("*.log"))
    
    print(f"📄 Found {len(log_files)} log files:")
    for log_file in sorted(log_files):
        print(f"  📄 {log_file.name}")
    
    print("\n" + "="*60)
    
    # Monitor files
    file_positions = {str(f): f.stat().st_size for f in log_files}
    
    try:
        while True:
            for log_file in log_files:
                current_size = log_file.stat().st_size
                last_size = file_positions.get(str(log_file), 0)
                
                if current_size > last_size:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        if new_content.strip():
                            print(f"\n📄 {log_file.name}:")
                            print(new_content, end='')
                    
                    file_positions[str(log_file)] = current_size
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Log monitoring stopped.")

def show_log_summary(log_dir: str = "logs"):
    """Show a summary of log files"""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"❌ Log directory {log_dir} does not exist!")
        return
    
    print(f"📊 Log Summary for: {log_path.absolute()}")
    print("="*60)
    
    log_files = list(log_path.glob("*.log"))
    
    if not log_files:
        print("📭 No log files found.")
        return
    
    total_size = 0
    for log_file in sorted(log_files):
        size = log_file.stat().st_size
        total_size += size
        modified = datetime.fromtimestamp(log_file.stat().st_mtime)
        
        print(f"📄 {log_file.name}")
        print(f"   Size: {size:,} bytes ({size/1024:.1f} KB)")
        print(f"   Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    print(f"📊 Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print(f"📊 Total files: {len(log_files)}")

def tail_log(log_file: str, lines: int = 50):
    """Show the last N lines of a log file"""
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ Log file {log_file} does not exist!")
        return
    
    print(f"📄 Last {lines} lines of {log_file}:")
    print("="*60)
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            for line in last_lines:
                print(line.rstrip())
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "monitor":
            follow = "--follow" in sys.argv
            monitor_logs(follow=follow)
        elif command == "summary":
            show_log_summary()
        elif command == "tail" and len(sys.argv) > 2:
            log_file = sys.argv[2]
            lines = int(sys.argv[3]) if len(sys.argv) > 3 else 50
            tail_log(log_file, lines)
        else:
            print("Usage:")
            print("  python monitor_logs.py monitor [--follow]")
            print("  python monitor_logs.py summary")
            print("  python monitor_logs.py tail <log_file> [lines]")
    else:
        print("🔍 Trading Bot Log Monitor")
        print("="*30)
        print("Commands:")
        print("  monitor [--follow]  - Monitor logs in real-time")
        print("  summary            - Show log file summary")
        print("  tail <file> [lines] - Show last N lines of a log file")
        print()
        print("Examples:")
        print("  python monitor_logs.py monitor")
        print("  python monitor_logs.py summary")
        print("  python monitor_logs.py tail logs/trading_bot_20240918.log 100")
