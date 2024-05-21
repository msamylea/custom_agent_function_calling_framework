import os
from dotenv import load_dotenv
import ostc
from openai import OpenAI
import json
from datetime import datetime

load_dotenv()

actions = ["Get the result of multiplying 4 and 6 then scheduling a meeting with Johan to dicuss the results."]


model = OpenAI(base_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct/v1/", api_key=os.environ.get("HF_TOKEN"))


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    result = a * b
    print(f"The result of multiplying {a} and {b} is {result}")
    return a * b

def create_meeting(attendee, time, description=None):
    
    print(f"Scheduled a meeting with {attendee} at {time}. {description}")

tools = [
        { 
            "name": "multiply",
            "description": "multiple two integers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "int",
                        "description": "The first integer to multiply"
                    },
                    "b": {
                        "type": "int",
                        "description": "The second integer to multiply"
                    }
                },
                "required": ["a", "b"]
            }
        },
        {
            "name": "create_meeting",
            "description": "Schedule a meeting for the user with the specified details",
            "parameters": {
                "type": "object",
                "properties": {
                    "attendee": {
                        "type": "string",
                        "description": "The person to schedule the meeting with"
                    },
                    "time": {
                        "type": "datetime",
                        "description": "The date and time of the meeting"
                    }
                },
                "required": [
                    "attendee",
                    "time"
                ]
            },
        },
]
multiply_tool = next(tool for tool in tools if tool["name"] == "multiply")
create_meeting_tool = next(tool for tool in tools if tool["name"] == "create_meeting")

functions = {
    "multiply": multiply,
    "create_meeting": create_meeting,
}

agent = ostc.AgentCreator(name="tool_user", tools=[multiply_tool],  description="Can use multiply tool to multiply two integers")
meeting_agent = ostc.AgentCreator(name="meeting_agent", tools=[create_meeting_tool],  description="Can use create_meeting tool to schedule a meeting")
extra_agent = ostc.AgentCreator(name="can't use tools", tools=[],  description="Can't use any tools")

agent.create_agent()
extra_agent.create_agent()
meeting_agent.create_agent()

agents = [agent, meeting_agent, extra_agent]

action_agents = ostc.AgentCreator.assign_agents(model, agents, actions)



def invoke_and_run(llm, action_agents):
    results = []
    for action, agent in action_agents.items():
        result = ostc.CallingFormat.generate_response(llm, agent.tools, actions)
        result = json.loads(result)
        function_name = result['tool']
        arguments = result.get('tool_input', {})
        results.append(functions[function_name](**arguments))
    return results


result = invoke_and_run(model, action_agents)
