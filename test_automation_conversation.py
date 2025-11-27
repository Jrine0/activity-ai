import os
import django
from django.test import RequestFactory
import json
import sys

# Setup Django environment
sys.path.append('c:/Users/jitin/Desktop/Desk/drive/Documents/all7')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_simulators.settings')
django.setup()

from sims.views import solve_automation_api

def run_conversation_test():
    factory = RequestFactory()
    
    # Initial State
    state = {
        'lights_living_room': {'status': 'off', 'brightness': 0},
        'thermostat': {'temperature': 70, 'mode': 'off'},
        'lock_front_door': {'status': 'locked'}
    }
    
    conversation_log = []
    
    # Conversation Turns
    turns = [
        "Hello, who are you?",
        "Turn on the living room lights and set brightness to 50%.",
        "What is the current temperature?",
        "Who won the 1998 World Cup?", # Guard rail test
        "Lock the front door."
    ]
    
    print("Starting Conversation Test...\n")
    
    for user_msg in turns:
        print(f"User: {user_msg}")
        
        # Prepare Request
        data = {
            'message': user_msg,
            'state': state
        }
        request = factory.post('/api/solve_automation/', data=json.dumps(data), content_type='application/json')
        
        # Call View
        try:
            response = solve_automation_api(request)
            result = json.loads(response.content)
            
            reply = result.get('reply', 'NO REPLY')
            updates = result.get('updates', {})
            error = result.get('error')
            
            print(f"Bot: {reply}")
            if updates:
                print(f"Updates: {json.dumps(updates, indent=2)}")
                # Update local state simulation
                for device, change in updates.items():
                    if device in state:
                        state[device].update(change)
            if error:
                print(f"ERROR: {error}")
                conversation_log.append(f"Error: {error}")
            
            # Log to file content
            conversation_log.append(f"User: {user_msg}")
            conversation_log.append(f"Bot: {reply}")
            if updates:
                conversation_log.append(f"Updates: {json.dumps(updates)}")
            
            if 'reply' not in result or result['reply'] == 'AI response could not be parsed.':
                print(f"FULL RESULT: {json.dumps(result, indent=2)}")
                conversation_log.append(f"FULL RESULT: {json.dumps(result, indent=2)}")
            conversation_log.append("-" * 20)
            
        except Exception as e:
            print(f"EXCEPTION: {e}")
            conversation_log.append(f"EXCEPTION: {e}")

    # Write Result File
    with open('automation_test_results.txt', 'w', encoding='utf-8') as f:
        f.write("Home Automation API Test Results\n")
        f.write("================================\n\n")
        for line in conversation_log:
            f.write(line + "\n")
            
    print("\nTest Complete. Results saved to automation_test_results.txt")

if __name__ == '__main__':
    run_conversation_test()
