from pydantic import BaseModel, ConfigDict, field_validator, ValidationError
from typing import Optional, Dict, List

import json
import re

class CallingFormat(BaseModel):
    model_config = ConfigDict(extra='ignore')
    tool: str
    tool_input: Optional[Dict] = None

    @field_validator('tool_input', mode='before')
    def parse_tool_input(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self.tool = data.get("tool")
        self.tool_input = data.get("tool_input")

    @staticmethod
    def generate_response(llm, tools, user_prompt: str):
        tool_descriptions = [
            f"{tool['name']}: {tool['description']}. Required parameters: {', '.join(tool['parameters']['required'])}"
            for tool in tools
        ]
        tool_descriptions_str = "\n".join(tool_descriptions)

        response = llm.chat.completions.create(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=[
                {"role": "system", "content": f"DO NOT REPEAT YOUR INSTRUCTIONS OR REPLY WITH ANYTHING NOT JSON. You have access to these tools:\n{tool_descriptions_str}"},
                {"role": "system", "content": "To use a tool, provide a JSON object with the 'tool' key specifying the tool name and the 'tool_input' key containing a dictionary of ALL the required parameters."},
                {"role": "system", "content": "Your response should be a single JSON array, where each element is a JSON object representing a tool usage for the entire user prompt. Do not include any other text or explanations."},
                {"role": "system", "content": "IMPORTANT: Ensure that you provide ALL the required parameters for each tool usage. If a required parameter is missing, the tool usage will fail."},
                {"role": "system", "content": "Example JSON format for multiple tool usages: [{\"tool\": \"tool_name1\", \"tool_input\": {\"param1\": \"value1\", \"param2\": \"value2\"}}, {\"tool\": \"tool_name2\", \"tool_input\": {\"param3\": \"value3\"}}]"},
                {"role": "system", "content": "If a tool does not require any input parameters, provide an empty dictionary for 'tool_input', like this: {\"tool\": \"tool_name\", \"tool_input\": {}}"},
                {"role": "user", "content": f"Based on the entire prompt '{user_prompt}', please provide a single JSON array containing the appropriate tool usages. Fill in ALL the required parameters for each tool based on the prompt. If a required parameter is not provided in the prompt, make a reasonable assumption or leave it empty, but ensure that all required parameters are present in the JSON. If no specific tool is required for a part, do not include a JSON object for that part. Follow the example JSON format strictly and do not include any non-JSON text."},
            ],
            max_tokens=4096,
            top_p=0.5,
            temperature=0.5,
            stop=["\n"],
        )
        response_text = response.choices[0].message.content.strip()
        if not response_text:
            return "The response from the language model is empty."

        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError:
            return f"The response from the language model is not a valid JSON string: {response_text}"

        valid_responses = []
        for resp in response_json:
            try:
                valid_data = CallingFormat.model_validate_json(json.dumps(resp))
                if valid_data:
                    valid_responses.append(resp)
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Error parsing or validating JSON: {resp}")
                print(f"Error message: {str(e)}")

        return valid_responses
class AgentCreator(BaseModel):
    name: str
    description: str
    tools: Optional[List[Dict]] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.name = data.get("name")
        self.description = data.get("description")
        self.tools = data.get("tools")

    def create_agent(self):
        return self
    
    def add_tools(self, tools):
        self.tools = tools
        return self
    @staticmethod
    def assign_agents(llm, agents, actions, max_retries=3):
        action_agents = {}

        # Make a single LLM call to determine the individual actions
        split_actions = llm.chat.completions.create(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=[
                {"role": "system", "content": f"Please provide a comma-separated list of individual actions from the following: {actions}. Do not include any other text."},
            ],
            max_tokens=4096,
            top_p=0.5,
            temperature=0.5,
            stop=["\n"],
        )

        split_actions = split_actions.choices[0].message.content.split(",")

        retries = 0
        while retries < max_retries:
            try:
                # Make a single LLM call to determine the appropriate tool for each action
                action_tool_mapping = llm.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-70B-Instruct",
                    messages=[
                        {"role": "system", "content": f"For each action in the following list, provide the corresponding tool to use: {split_actions}. The available tools are: {[tool['name'] for agent in agents for tool in agent.tools]}. Reply with a JSON object where the keys are the actions enclosed in double quotes and the values are the tool names enclosed in double quotes. For example: {{\"action1\": \"tool1\", \"action2\": \"tool2\"}}. Do not include any other text. Do not use the chat response if another tool can perform the action. If a chat response can be combined with another action (for example a chat at a meeting), it should go with the non-chat tool. You should combine as much as possible. If you have two create_meeting for example, combine them into one create_meeting"},
                    ],
                    max_tokens=4096,
                    top_p=0.5,
                    temperature=0.5,
                    stop=["\n"],
                )

                action_tool_mapping = json.loads(action_tool_mapping.choices[0].message.content)
                break  # Exit the loop if parsing is successful
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {str(e)}")
                retries += 1
                if retries == max_retries:
                    raise ValueError("Failed to parse action-tool mapping after multiple retries.")

        for action, tool_name in action_tool_mapping.items():
            for agent in agents:
                if agent.tools:
                    # Check if the agent has the appropriate tool for the action
                    if any(tool["name"] == tool_name for tool in agent.tools):
                        action_agents[action] = agent
                        break
        print(action_tool_mapping)                
        return action_agents