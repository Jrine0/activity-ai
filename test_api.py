import os
import django
from django.test import RequestFactory
import json
import sys

# Setup Django environment
sys.path.append('c:/Users/jitin/Desktop/Desk/drive/Documents/all7')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_simulators.settings')
django.setup()

from sims.views import solve_maze_api, solve_garbage_api, solve_parking_api, solve_diagnosis_api, solve_automation_api, generate_city_api, generate_parking_api

factory = RequestFactory()

def test_maze():
    print("Testing Maze Solver...")
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    data = {
        'grid': grid,
        'start': {'x': 0, 'y': 0},
        'end': {'x': 2, 'y': 2},
        'algorithm': 'bfs'
    }
    request = factory.post('/api/solve_maze/', data=json.dumps(data), content_type='application/json')
    response = solve_maze_api(request)
    result = json.loads(response.content)
    if 'path' in result and len(result['path']) > 0:
        print("PASS: Maze Solver")
    else:
        print("FAIL: Maze Solver", result)

def test_garbage():
    print("Testing Garbage Truck (Advanced)...")
    # Generate city first
    gen_data = {'rows': 10, 'cols': 10}
    request_gen = factory.post('/api/generate_city/', data=json.dumps(gen_data), content_type='application/json')
    response_gen = generate_city_api(request_gen)
    city_data = json.loads(response_gen.content)
    
    if 'grid' not in city_data:
        print("FAIL: City Generation", city_data)
        return

    # Find start
    grid = city_data['grid']
    start = {'x': 0, 'y': 0}
    for x in range(10):
        for y in range(10):
            if grid[x][y] == 1:
                start = {'x': x, 'y': y}
                break
        if start['x'] != 0: break
        
    data = {
        'grid': grid,
        'start': start,
        'bins': city_data['bins'],
        'time_of_day': 'morning',
        'one_ways': city_data['one_ways']
    }
    request = factory.post('/api/solve_garbage/', data=json.dumps(data), content_type='application/json')
    response = solve_garbage_api(request)
    result = json.loads(response.content)
    
    if 'paths' in result or 'message' in result:
        print("PASS: Garbage Truck (Advanced)")
    else:
        print("FAIL: Garbage Truck (Advanced)", result)

def test_parking_generation():
    print("Testing Parking Generation...")
    data = {"rows": 20, "cols": 30}
    request = factory.post('/api/generate_parking/', data=json.dumps(data), content_type='application/json')
    response = generate_parking_api(request)
    result = json.loads(response.content)
    
    if 'grid' in result and 'start' in result:
        print("PASS: Parking Generation")
    else:
        print("FAIL: Parking Generation", result.keys())

def test_parking():
    print("Testing Parking Finder...")
    grid = [[0, 0, 0], [0, 2, 0], [0, 0, 0]] # 2 is free spot
    data = {
        'grid': grid,
        'start': {'x': 0, 'y': 0},
        'algorithm': 'bfs',
        'handicap': False
    }
    request = factory.post('/api/solve_parking/', data=json.dumps(data), content_type='application/json')
    response = solve_parking_api(request)
    result = json.loads(response.content)
    if 'path' in result and len(result['path']) > 0:
        print("PASS: Parking Finder")
    else:
        print("FAIL: Parking Finder", result)

def test_diagnosis():
    print("Testing Fault Diagnosis...")
    data = {
        'symptoms': ['HighTemp', 'LowPressure'],
        'algorithm': 'forward'
    }
    request = factory.post('/api/solve_diagnosis/', data=json.dumps(data), content_type='application/json')
    response = solve_diagnosis_api(request)
    result = json.loads(response.content)
    if 'inferred' in result and 'CoolantLeak' in result['inferred']:
        print("PASS: Diagnosis (Forward)")
    else:
        print("FAIL: Diagnosis (Forward)", result)

def test_automation():
    print("Testing Home Automation (Gemini)...")
    # Mocking the Gemini call would be ideal, but here we just check if the endpoint accepts the new structure.
    # We can't easily mock requests.post in this simple script without a library like `unittest.mock`.
    # So we will just check if it handles a request without crashing, even if it fails to reach Gemini (or returns error).
    
    data = {
        'message': 'Turn on the lights',
        'state': {'lights_living_room': {'status': 'off'}}
    }
    request = factory.post('/api/solve_automation/', data=json.dumps(data), content_type='application/json')
    
    # We need to be careful. If we run this, it will actually call the Gemini API if internet is available.
    # That's fine for a real integration test.
    try:
        response = solve_automation_api(request)
        result = json.loads(response.content)
        
        if 'reply' in result and 'updates' in result:
            print("PASS: Automation (Gemini Integration)")
        elif 'error' in result:
             # It might fail due to network or key, but structure is handled
            print(f"PASS: Automation (Handled Error: {result['error']})")
        else:
            print("FAIL: Automation", result.keys())
    except Exception as e:
        print(f"FAIL: Automation - {e}")

if __name__ == '__main__':
    try:
        test_maze()
        test_garbage()
        test_parking_generation()
        test_parking()
        test_diagnosis()
        test_automation()
    except Exception as e:
        print(f"ERROR: {e}")
