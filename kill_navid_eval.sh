ps aux | grep 'run_vision_info.py' | grep 'navid'   | awk '{print $2}' | xargs kill
