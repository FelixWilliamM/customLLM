
# app.py
import os
import json
import logging
from flask import Flask, Blueprint, request, Response, jsonify
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
custom_llm = Blueprint('custom_llm', __name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# File paths
CALL_STATE_FILE = 'call_states.json'
PATHWAYS_MESSAGES_FILE = 'pathways.json'
ASSISTANT_CONFIG_FILE = 'assistant_config.json'

# Ensure the call state file exists
if not os.path.exists(CALL_STATE_FILE):
    with open(CALL_STATE_FILE, 'w') as f:
        json.dump({}, f)

# Ensure the assistant config file exists with default configuration
if not os.path.exists(ASSISTANT_CONFIG_FILE):
    default_config = {
        "model": {
            "provider": "openai",
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an assistant. When the user asks to be transferred, use the transferCall function."
                }
            ]
        },
        "forwardingPhoneNumber": "+40761983263"
    }
    with open(ASSISTANT_CONFIG_FILE, 'w') as f:
        json.dump(default_config, f, indent=2)

# Ensure the pathways file exists with a default structure
if not os.path.exists(PATHWAYS_MESSAGES_FILE):
    default_pathways = [
        {
            "name": "start",
            "block": {
                "instruction": "You are a helpful assistant. Help users with their questions and transfer them when requested."
            },
            "destinations": [
                {
                    "stepName": "start"
                }
            ]
        }
    ]
    with open(PATHWAYS_MESSAGES_FILE, 'w') as f:
        json.dump(default_pathways, f, indent=2)

def load_assistant_config():
    with open(ASSISTANT_CONFIG_FILE, 'r') as f:
        return json.load(f)

def load_pathways():
    with open(PATHWAYS_MESSAGES_FILE, 'r') as f:
        raw_nodes = json.load(f)
    return {node["name"]: node for node in raw_nodes}

# Load configurations
assistant_config = load_assistant_config()
flow_nodes = load_pathways()

def load_call_states():
    with open(CALL_STATE_FILE, 'r') as f:
        return json.load(f)

def save_call_states(states):
    with open(CALL_STATE_FILE, 'w') as f:
        json.dump(states, f)

def get_current_node_name(call_id):
    states = load_call_states()
    node_name = states.get(call_id)

    if not node_name:
        node_name = "start" if "start" in flow_nodes else list(flow_nodes.keys())[0]
        states[call_id] = node_name
        save_call_states(states)

    return node_name

def set_current_node_name(call_id, node_name):
    states = load_call_states()
    states[call_id] = node_name
    save_call_states(states)

def get_available_functions():
    functions = []

    if assistant_config.get("forwardingPhoneNumber"):
        functions.append({
            "name": "transferCall",
            "description": "Transfer the call to another phone number",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The destination phone number to transfer the call to"
                    }
                },
                "required": ["destination"]
            }
        })

    return functions

def generate_streaming_response(stream_response):
    for chunk in stream_response:
        if chunk.choices[0].delta.content is not None:
            data = json.dumps({"choices": [{"delta": {"content": chunk.choices[0].delta.content}}]})
            yield f"data: {data}\n\n"

        if hasattr(chunk.choices[0].delta, 'function_call') and chunk.choices[0].delta.function_call:
            function_call = chunk.choices[0].delta.function_call
            if hasattr(function_call, 'name') and function_call.name == "transferCall":
                try:
                    arguments = json.loads(function_call.arguments)
                    destination = arguments.get("destination", assistant_config["forwardingPhoneNumber"])
                    function_call_payload = {
                        "function_call": {
                            "name": "transferCall",
                            "arguments": {
                                "destination": destination
                            }
                        }
                    }
                    yield f"data: {json.dumps(function_call_payload)}\n\n"
                except json.JSONDecodeError:
                    logger.error("Failed to parse transferCall arguments")

@custom_llm.route('/chat/completions', methods=['POST'])
def openai_advanced_custom_llm_route():
    request_data = request.get_json()
    streaming = request_data.get('stream', False)
    call_id = request_data['call']['id']

    current_node_name = get_current_node_name(call_id)
    current_node = flow_nodes[current_node_name]

    block_data = current_node.get("block", {})
    next_prompt = block_data.get("instruction", "No instructions available.")

    system_message = next_prompt
    if assistant_config["forwardingPhoneNumber"]:
        system_message += "\nIf the user requests a transfer, you can transfer them using the transferCall function."

    messages = [{"role": "system", "content": system_message}]

    if 'messages' in request_data and request_data['messages']:
        messages.append(request_data['messages'][-1])

    openai_params = {
        "model": assistant_config["model"]["model"],
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.2,
        "stream": streaming
    }

    functions = get_available_functions()
    if functions:
        openai_params["functions"] = functions
        openai_params["function_call"] = "auto"

    destinations = current_node.get("destinations", [])
    if destinations:
        next_node_name = destinations[0].get("stepName")
        if next_node_name and next_node_name in flow_nodes:
            set_current_node_name(call_id, next_node_name)

    if streaming:
        completion_stream = client.chat.completions.create(**openai_params)
        return Response(
            generate_streaming_response(completion_stream),
            content_type='text/event-stream'
        )
    else:
        completion = client.chat.completions.create(**openai_params)
        return jsonify(completion)

@custom_llm.route('/config', methods=['GET', 'POST'])
def config():
    if request.method == 'GET':
        return jsonify(assistant_config)

    new_config = request.get_json()
    if new_config:
        assistant_config.update(new_config)
        with open(ASSISTANT_CONFIG_FILE, 'w') as f:
            json.dump(assistant_config, f, indent=2)
        return jsonify({"status": "success", "message": "Configuration updated"})

    return jsonify({"status": "error", "message": "Invalid configuration"}), 400

@app.route('/')
def home():
    return "Hello from Flask! The server is running."

app.register_blueprint(custom_llm)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
