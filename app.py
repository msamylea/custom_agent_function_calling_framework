import ostc
import config as cfg


actions = ["Get the result of multiplying 4 and 6 then scheduling a meeting with Johan at a date and time of your choice. Next, tell me a fact about penguins."]


model = cfg.llm

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    result = a * b
    print(f"The result of multiplying {a} and {b} is {result}")
    return a * b

def create_meeting(attendee, time, description=None):
    
    print(f"Scheduled a meeting with {attendee} at {time}.")

def chat_response(response: str):
    print(response)

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
        {
            "name": "chat_response",
            "description": (
                "Use chat responses if a conversation is appropriate instead of a function."
        ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Standard chat response.",
                    },
                },
                "required": ["response"],
        },
}
]
multiply_tool = next((tool for tool in tools if tool["name"] == "multiply"), None)
create_meeting_tool = next((tool for tool in tools if tool["name"] == "create_meeting"), None)
chat_response_tool = next((tool for tool in tools if tool["name"] == "chat_response"), None)

functions = {
    "multiply": multiply,
    "create_meeting": create_meeting,
    "chat_response": chat_response

}

agent_creators = [
    ostc.AgentCreator(name="tool_user", tools=[multiply_tool], description="Can use multiply tool to multiply two integers"),
    ostc.AgentCreator(name="meeting_agent", tools=[create_meeting_tool], description="Can use create_meeting tool to schedule a meeting"),
    ostc.AgentCreator(name="can_chat", tools=[chat_response_tool], description="Can use chat responses to respond to user prompts"),

]

agents = [creator.create_agent() for creator in agent_creators]

action_agents = ostc.AgentCreator.assign_agents(model, agents, actions)


def invoke_and_run(llm, action_agents):
    results = []
    for action, agent in action_agents.items():
        result = ostc.CallingFormat.generate_response(llm, agent.tools, action)
        if isinstance(result, str):
            print(f"Error: {result}")
            continue
        for res in result:
            function_name = res['tool']
            arguments = res.get('tool_input', {})
            if function_name in functions:
                results.append(functions[function_name](**arguments))
            else:
                print(f"Error: Unknown function '{function_name}'")
    return results

result = invoke_and_run(model, action_agents)