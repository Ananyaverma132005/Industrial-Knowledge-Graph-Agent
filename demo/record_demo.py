"""Record real agent responses for the static GitHub Pages demo.

Runs the FastAPI app in-process (same code path as `uvicorn main:app`) and
saves its actual responses to demo/recorded.json:
  * /health, /graph-data, /predict-maintenance and /node/<id> for every node
  * /chat answers for a fixed set of questions (including a follow-up that
    uses conversation memory)
  * a stream of telemetry frames from the same simulator the WebSocket uses

Usage (from the project root, after setup.ps1 / ingest):
    python demo/record_demo.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from telemetry import TelemetrySimulator  # noqa: E402

OUT = os.path.join(ROOT, 'demo', 'recorded.json')
TELEMETRY_FRAMES = 60

MACHINES = ['CNC milling machine', 'robotic welding arm', 'hydraulic press',
            'belt conveyor', 'CNC lathe', 'pick-and-place robot']
PARTS = ['spindle', 'hydraulic seal', 'drive belt', 'servo motor', 'gearbox',
         'coolant pump', 'bearing', 'welding torch', 'gripper']

QUESTIONS = [
    'Which parts are at risk of failure?',
    *[f'What parts does the {m} have?' for m in MACHINES],
    *[f'Who maintains the {m}?' for m in MACHINES],
    *[f'When was the {m} last serviced?' for m in MACHINES],
    *[f'Show failure tickets for the {p}' for p in PARTS],
    'How do I fix spindle overheating?',
    'Procedure to replace a hydraulic seal',
    'What is the lockout tagout procedure?',
    'How do I fix conveyor belt misalignment?',
    'How do I maintain the welding torch?',
    'What should I do if a servo motor overheats?',
    'How do I troubleshoot a coolant pump failure?',
    'How do I inspect the lathe chuck?',
    'What to do when the gripper drops parts?',
    'What PPE is required on the assembly line?',
    'What is the capital of France?',
]

# Follow-ups asked right after a question, to show conversation memory.
FOLLOW_UPS = [
    ('Who maintains the hydraulic press?', 'When was it last serviced?'),
    ('What parts does the CNC milling machine have?', 'Which of them have failure tickets?'),
]


def main_record():
    data = {'chat': []}
    with TestClient(main.app) as client:
        data['health'] = client.get('/health').json()
        data['graph'] = client.get('/graph-data').json()
        data['risk'] = client.get('/predict-maintenance').json()

        nodes = {}
        for n in data['graph']['nodes']:
            r = client.get(f"/node/{n['id']}")
            if r.status_code == 200:
                nodes[n['id']] = r.json()
        data['nodes'] = nodes

        for q in QUESTIONS:
            r = client.post('/chat', json={'query': q}).json()
            r.pop('session_id', None)
            data['chat'].append({'q': q, 'prev': None, 'response': r})
            print('recorded:', q, '->', r.get('route'))

        for first, follow in FOLLOW_UPS:
            sid = client.post('/chat', json={'query': first}).json()['session_id']
            r = client.post('/chat', json={'query': follow, 'session_id': sid}).json()
            r.pop('session_id', None)
            data['chat'].append({'q': follow, 'prev': first, 'response': r})
            print('recorded follow-up:', follow)

        sim = TelemetrySimulator(main.state['store'].nodes('Machine'), main.state['model'], seed=7)
        data['telemetry'] = [sim.tick() for _ in range(TELEMETRY_FRAMES)]

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    print(f'Saved {len(data["chat"])} answers, {len(nodes)} nodes, '
          f'{TELEMETRY_FRAMES} telemetry frames to {OUT}')


if __name__ == '__main__':
    main_record()
