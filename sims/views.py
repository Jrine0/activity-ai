from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import requests

from .algorithms import (
    solve_maze_algo, solve_garbage_algo, solve_parking_algo,
    solve_diagnosis_algo, generate_city_map, generate_parking_lot
)

# -----------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------

def safe_load_json(request):
    """Safely parse request.body as JSON."""
    try:
        return json.loads(request.body)
    except Exception:
        return None

def get_position(data, key):
    """Convert {x: ?, y: ?} to (x, y)."""
    pos = data.get(key)
    if not pos or "x" not in pos or "y" not in pos:
        return None
    return (pos["x"], pos["y"])

# -----------------------------------------------
# PAGE RENDERERS
# -----------------------------------------------

def landing_page(request):
    return render(request, 'sims/index.html')

def maze_solver(request):
    return render(request, 'sims/maze.html')

def garbage_truck(request):
    return render(request, 'sims/garbage.html')

def parking_finder(request):
    return render(request, 'sims/parking.html')

def fault_diagnosis(request):
    return render(request, 'sims/diagnosis.html')

def home_automation(request):
    return render(request, 'sims/automation.html')

# -----------------------------------------------
# API ENDPOINTS
# -----------------------------------------------

@csrf_exempt
def solve_maze_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=400)

    data = safe_load_json(request)
    if not data:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    grid = data.get('grid')
    start = get_position(data, 'start')
    end = get_position(data, 'end')
    algo = data.get('algorithm')

    if not (grid and start and end and algo):
        return JsonResponse({'error': 'Missing fields'}, status=400)

    return JsonResponse(solve_maze_algo(grid, start, end, algo))


@csrf_exempt
def solve_garbage_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=400)

    data = safe_load_json(request)
    if not data:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    grid = data.get('grid')
    start = get_position(data, 'start')
    bins = data.get('bins', [])
    time_of_day = data.get('time_of_day', 'night')
    one_ways = data.get('one_ways', [])

    if not (grid and start):
        return JsonResponse({'error': 'Missing fields'}, status=400)

    return JsonResponse(solve_garbage_algo(grid, start, bins, time_of_day, one_ways))


@csrf_exempt
def generate_city_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=400)

    data = safe_load_json(request) or {}
    rows = data.get('rows', 20)
    cols = data.get('cols', 30)

    return JsonResponse(generate_city_map(rows, cols))


@csrf_exempt
def solve_parking_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=400)

    data = safe_load_json(request)
    if not data:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    grid = data.get('grid')
    start = get_position(data, 'start')
    algo = data.get('algorithm')
    handicap = data.get('handicap')

    if not (grid and start and algo):
        return JsonResponse({'error': 'Missing fields'}, status=400)

    return JsonResponse(solve_parking_algo(grid, start, algo, handicap))


@csrf_exempt
def generate_parking_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=400)

    data = safe_load_json(request) or {}
    rows = data.get('rows', 20)
    cols = data.get('cols', 30)

    return JsonResponse(generate_parking_lot(rows, cols))


@csrf_exempt
def solve_diagnosis_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=400)

    data = safe_load_json(request)
    if not data:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    symptoms = data.get('symptoms', [])
    algo = data.get('algorithm')
    target = data.get('target')

    if not algo:
        return JsonResponse({'error': 'Algorithm missing'}, status=400)

    return JsonResponse(solve_diagnosis_algo(symptoms, algo, target))

# ------------------------------------------------------
# ⭐ SMART HOME CHATBOT API (NEW FULLY WORKING VERSION)
# ------------------------------------------------------

@csrf_exempt
def solve_automation_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=400)

    data = safe_load_json(request)
    if not data:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    user_message = data.get("message", "").strip()
    current_state = data.get("state", {})

    API_KEY = getattr(settings, "GEMINI_API_KEY", None)
    if not API_KEY:
        return JsonResponse({'error': 'Missing Gemini API key'}, status=500)

    URL = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.0-flash:generateContent?key={API_KEY}"
    )

    # -------------------------
    # CHATBOT SYSTEM PROMPT
    # -------------------------
    system_prompt = (
        "You are a Smart Home Assistant AI. Your ONLY function is to operate "
        "and report on home automation devices.\n\n"
        "STRICT RULES:\n"
        "1. If user asks anything unrelated to home automation (math, history, "
        "politics, facts), REFUSE with: "
        "\"I can only help with smart home controls.\"\n"
        "2. For unsafe actions (unlocking doors at night), warn the user but "
        "still execute.\n"
        "3. ALWAYS respond ONLY with valid JSON:\n"
        "{\"reply\": \"...\", \"updates\": {...}}\n"
        "4. Never output code blocks or markdown.\n\n"
        f"Current state:\n{json.dumps(current_state)}\n\n"
        f"User message: \"{user_message}\"\n"
        "Respond only using this JSON format."
    )

    payload = {
        "contents": [
            {
                "parts": [{"text": system_prompt}]
            }
        ]
    }

    # -------------------------
    # CALL GEMINI
    # -------------------------
    try:
        response = requests.post(
            URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        return JsonResponse({"reply": f"Network error: {str(e)}", "updates": {}})

    if response.status_code != 200:
        return JsonResponse({
            "reply": "AI service unavailable.",
            "updates": {},
            "error": response.text
        })

    # -------------------------
    # PARSE GEMINI RESPONSE
    # -------------------------
    try:
        gem = response.json()
        text = gem["candidates"][0]["content"]["parts"][0]["text"]

        # Remove accidental markdown
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1].strip()
                if text.lower().startswith("json"):
                    text = text[4:].strip()

        result = json.loads(text)
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({
            "reply": "AI response could not be parsed.",
            "updates": {},
            "debug_error": str(e),
            "raw_response": response.text
        })
