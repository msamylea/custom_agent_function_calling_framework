from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, List


   
CONVERSATION_FUNCTION = {
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

class CallingFormat(BaseModel):
    model_config = ConfigDict(extra='ignore')
    tool: str
    tool_input: Optional[Dict] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.tool = data.get("tool")
        self.tool_input = data.get("tool_input")

    @staticmethod
    def generate_response(llm, tools, user_prompt: str):
        response = llm.chat.completions.create(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant with access to these {tools} to answer the {user_prompt} question:"},
                {"role": "system", "content": f"If you do not need to use a tool, you can response with {CONVERSATION_FUNCTION}"},
                {"role": "system", "content": "You must reply in valid JSON format, with no other text. Example: {\"tool\": \"tool_name\", \"tool_input\": \"tool_input\"}"},
            ],
            max_tokens=4096,
            top_p=0.5,
            temperature=0.5,
            stop=["\n"],
        )
        response = response.choices[0].message.content
        valid_data = CallingFormat.model_validate_json(response)
        if valid_data:
            return response
        else:
            return "Please reply in valid JSON format."
    

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
    def assign_agents(llm, agents, actions):
        action_agents = {}
        for agent in agents:
            if agent.tools:
                split_actions =  llm.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-70B-Instruct",
                    messages=[
                        {"role": "system", "content": f"You are tasked with determining the individual actions from the {actions}. There may be multiple actions to determine or only one."},
                        {"role": "system", "content": f"You have a list of {agent.tools} to use to help determine which actions are individual actions."},
                        {"role": "system", "content": f"You must reply with only the name of the {actions} to perform. Do not reply with any other text."},
                    ],
                    max_tokens=4096,
                    top_p=0.5,
                    temperature=0.5,
                    stop=["\n"],
                )
                for choice in split_actions.choices:
                   split_actions= choice.message.content
                if split_actions:
                        for action in split_actions:
                            selected_tool =  llm.chat.completions.create(
                            model="meta-llama/Meta-Llama-3-70B-Instruct",
                            messages=[
                                {"role": "system", "content": f"You are tasked with determining the best tools to use to answer the question: {action}."},
                                {"role": "system", "content": f"After you have determined the best tools to use, you must determine which {agent.tools} to use to answer the question."},
                                {"role": "system", "content": f"You must reply with only the name of the tool you would use from {agent.tools} with the specific section of the {action} to perform. Do not reply with any other text."},
                            ],
                            max_tokens=4096,
                            top_p=0.5,
                            temperature=0.5,
                            stop=["\n"],
                        )
                        response = selected_tool.choices[0].message.content
                        if response in [tool['name'] for tool in agent.tools]:
                            action_agents[str(action)] = agent
                            action_agents[action] = agent
        return action_agents


